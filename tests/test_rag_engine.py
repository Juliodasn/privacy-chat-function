import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AvaliacaoDataMapping.rag_engine import (  # noqa: E402
    RagChunk,
    build_chunks,
    evaluate_questions_with_rag,
    retrieve_context_for_question,
)


class RagEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prev_openai = os.environ.get("OPENAI_API_KEY")
        self.prev_azure_a = os.environ.get("AZURE_OPENAI_API_KEY")
        self.prev_azure_b = os.environ.get("AZURE_OPENAI_KEY")
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_KEY"] = ""

    def tearDown(self) -> None:
        def _restore(key: str, value: str | None) -> None:
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        _restore("OPENAI_API_KEY", self.prev_openai)
        _restore("AZURE_OPENAI_API_KEY", self.prev_azure_a)
        _restore("AZURE_OPENAI_KEY", self.prev_azure_b)

    def test_build_chunks_bounds(self) -> None:
        long_text = ("Linha de teste para chunking.\n" * 200).strip()
        chunks = build_chunks(long_text, {"__raw_text_src__": long_text}, chunk_size=1200, overlap=120)
        self.assertGreaterEqual(len(chunks), 2)
        for ch in chunks:
            self.assertLessEqual(len(ch.text), 1500)

    def test_retrieve_context_respects_section_filter(self) -> None:
        index = {
            "has_embeddings": True,
            "chunks": [
                RagChunk("a1", "areas", "table:areas", "contexto area", {}),
                RagChunk("p1", "processos", "table:processos", "contexto processo", {}),
            ],
            "embeddings": [[1.0, 0.0], [0.0, 1.0]],
            "section_positions": {"areas": [0], "processos": [1]},
            "question_sections": {"q_area": "areas"},
            "_question_embedding_cache": {"q_area::pergunta area": [1.0, 0.0]},
            "_embedding_client": None,
            "embedding_model": None,
        }
        hits = retrieve_context_for_question("q_area", "pergunta area", index, top_k=3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["chunk_id"], "a1")
        self.assertEqual(hits[0]["section"], "areas")

    def test_evaluate_questions_with_rag_returns_metrics_without_chat_client(self) -> None:
        index = {
            "has_embeddings": True,
            "chunks": [RagChunk("a1", "areas", "table:areas", "contexto area", {})],
            "embeddings": [[1.0, 0.0]],
            "section_positions": {"areas": [0]},
            "question_sections": {"q1": "areas"},
            "_question_embedding_cache": {"q1::pergunta area": [1.0, 0.0]},
            "_embedding_client": None,
            "embedding_model": None,
        }
        out = evaluate_questions_with_rag(
            [("q1", "pergunta area")],
            index,
            top_k=4,
            batch_size=6,
            max_context_chars=4000,
            max_context_tokens=1000,
        )
        self.assertIn("results", out)
        self.assertIn("meta", out)
        self.assertEqual(out["meta"]["llm_calls"], 0)
        self.assertIn("retrieval_ms_total", out["meta"])
        self.assertEqual(out["results"]["q1"]["reason"], "no_chat_client")
        self.assertEqual(out["results"]["q1"]["rag_chunks"][0]["chunk_id"], "a1")


if __name__ == "__main__":
    unittest.main()
