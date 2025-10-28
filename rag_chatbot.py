"""Simple Retrieval-Augmented Generation (RAG) chatbot for admissions advising.

This script loads knowledge from PDF handbooks and FAQ Excel sheets, builds a
light-weight TF-IDF retrieval index, and surfaces answers to admissions
questions. It is designed as an approachable baseline that runs locally without
external services.  The chatbot returns answers directly from FAQ spreadsheets
when available and otherwise quotes the most relevant passages from the PDF
documents.

Usage example::

    python rag_chatbot.py --pdf-dir data/pdfs --excel-file data/faq.xlsx

Once the knowledge base is loaded, an interactive prompt is shown. Type your
question and press enter to receive an answer. Enter ``exit`` or ``quit`` to
leave the chat session.
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class Document:
    """A knowledge base entry used by the chatbot."""

    content: str
    source: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Stores the result of retrieving knowledge for a user query."""

    answer: str
    sources: List[Dict[str, str]]


class SimpleRAGChatbot:
    """A minimal Retrieval-Augmented Generation chatbot.

    The implementation uses TF-IDF bag-of-words vectors for retrieval and a
    deterministic, template-based response generator. The goal is to keep the
    system easy to understand and simple to deploy.
    """

    def __init__(self, documents: Sequence[Document], top_k: int = 3):
        if not documents:
            raise ValueError("At least one document is required to initialize the chatbot.")

        self.documents = list(documents)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(stop_words="english")
        logging.info("Building TF-IDF index for %d documents...", len(self.documents))
        self.matrix = self.vectorizer.fit_transform(doc.content for doc in self.documents)

    def _format_source(self, doc: Document, score: float) -> Dict[str, str]:
        snippet = textwrap.shorten(doc.content, width=280, placeholder="...")
        return {
            "source": doc.source,
            "score": f"{score:.3f}",
            "snippet": snippet,
        }

    def _generate_answer(self, query: str, doc: Document) -> str:
        if doc.metadata.get("type") == "qa":
            return doc.metadata.get("answer", doc.content)

        header = f"Dưới đây là thông tin liên quan từ tài liệu {doc.source}:"
        paragraph = textwrap.fill(doc.content, width=90)
        return f"{header}\n\n{paragraph}"

    def query(self, question: str) -> RetrievalResult:
        logging.info("Retrieving knowledge for the incoming question.")
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        if not np.any(scores):
            return RetrievalResult(
                answer="Xin lỗi, tôi chưa tìm thấy thông tin phù hợp. Bạn có thể đặt câu hỏi khác không?",
                sources=[],
            )

        top_indices = scores.argsort()[::-1][: self.top_k]
        top_docs = [self.documents[idx] for idx in top_indices]
        top_scores = [scores[idx] for idx in top_indices]

        answer = self._generate_answer(question, top_docs[0])
        sources = [self._format_source(doc, score) for doc, score in zip(top_docs, top_scores)]
        return RetrievalResult(answer=answer, sources=sources)

    def chat(self) -> None:
        print("Chatbot RAG đã sẵn sàng! Gõ 'exit' hoặc 'quit' để kết thúc.")
        while True:
            try:
                question = input("\nBạn: ")
            except EOFError:
                print("\nTạm biệt!")
                break

            if not question:
                continue

            if question.strip().lower() in {"exit", "quit"}:
                print("Tạm biệt!")
                break

            result = self.query(question)
            print("\nChatbot:")
            print(result.answer)
            if result.sources:
                print("\nNguồn tham khảo:")
                for src in result.sources:
                    print(f"- {src['source']} (score={src['score']}): {src['snippet']}")


def read_pdf_documents(pdf_paths: Iterable[Path]) -> List[Document]:
    documents: List[Document] = []
    for path in pdf_paths:
        logging.info("Đang đọc PDF: %s", path)
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        content = "\n".join(page.strip() for page in pages if page)
        if content:
            documents.append(Document(content=content, source=path.name, metadata={"type": "pdf"}))
        else:
            logging.warning("Không thể trích xuất nội dung từ %s", path)
    return documents


def read_excel_qa(
    excel_files: Iterable[Path],
    question_column: str = "question",
    answer_column: str = "answer",
) -> List[Document]:
    documents: List[Document] = []
    for path in excel_files:
        logging.info("Đang đọc file QA: %s", path)
        df = pd.read_excel(path)
        normalized_columns = {col.lower().strip(): col for col in df.columns}

        if question_column not in normalized_columns:
            raise KeyError(
                f"Cột câu hỏi '{question_column}' không tồn tại trong {path}. Các cột hiện có: {list(df.columns)}"
            )
        if answer_column not in normalized_columns:
            raise KeyError(
                f"Cột trả lời '{answer_column}' không tồn tại trong {path}. Các cột hiện có: {list(df.columns)}"
            )

        q_col = normalized_columns[question_column]
        a_col = normalized_columns[answer_column]

        for _, row in df[[q_col, a_col]].dropna().iterrows():
            question = str(row[q_col]).strip()
            answer = str(row[a_col]).strip()
            if not question or not answer:
                continue

            documents.append(
                Document(
                    content=f"Q: {question}\nA: {answer}",
                    source=f"{path.name} - {question[:60]}",
                    metadata={"type": "qa", "question": question, "answer": answer},
                )
            )
    return documents


def build_chatbot(
    pdf_dir: Optional[Path],
    excel_files: Sequence[Path],
    question_column: str = "question",
    answer_column: str = "answer",
    top_k: int = 3,
) -> SimpleRAGChatbot:
    documents: List[Document] = []

    if pdf_dir is not None:
        pdf_paths = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_paths:
            logging.warning("Không tìm thấy file PDF trong thư mục %s", pdf_dir)
        documents.extend(read_pdf_documents(pdf_paths))

    if excel_files:
        documents.extend(
            read_excel_qa(
                excel_files,
                question_column=question_column.lower().strip(),
                answer_column=answer_column.lower().strip(),
            )
        )

    if not documents:
        raise ValueError("Không có dữ liệu để xây dựng chatbot. Kiểm tra đường dẫn đầu vào.")

    return SimpleRAGChatbot(documents, top_k=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chatbot RAG hỗ trợ tư vấn tuyển sinh")
    parser.add_argument("--pdf-dir", type=Path, help="Thư mục chứa các file PDF", default=None)
    parser.add_argument(
        "--excel-file",
        type=Path,
        action="append",
        help="Đường dẫn tới file Excel chứa cột câu hỏi và trả lời. Có thể truyền nhiều lần.",
    )
    parser.add_argument("--question-column", default="question", help="Tên cột câu hỏi trong Excel")
    parser.add_argument("--answer-column", default="answer", help="Tên cột câu trả lời trong Excel")
    parser.add_argument("--top-k", type=int, default=3, help="Số lượng nguồn tham khảo hiển thị")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    excel_files= "data/faq.xlsx"
    pdf_dir= "data/pdfs"

    chatbot = build_chatbot(
        pdf_dir=pdf_dir,
        excel_files=excel_files,
        question_column=args.question_column,
        answer_column=args.answer_column,
        top_k=args.top_k,
    )
    chatbot.chat()


if __name__ == "__main__":
    main()