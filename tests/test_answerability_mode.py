import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import AvaliacaoDataMapping as dm  # noqa: E402

try:
    import pandas as pd
except Exception:
    pd = None


class AnswerabilityModeTests(unittest.TestCase):
    def test_has_structured_tables_false_for_raw_text_only(self) -> None:
        structured = {"__raw_text__": "conteudo sem tabela", "__tables__": []}
        self.assertFalse(dm.has_structured_tables(structured))

    @unittest.skipIf(pd is None, "pandas not available")
    def test_has_structured_tables_true_for_dataframe(self) -> None:
        structured = {"__tables__": [("areas", pd.DataFrame([{"coluna": "valor"}]))]}
        self.assertTrue(dm.has_structured_tables(structured))

    def test_compute_answered_uses_rag_when_rag_primary(self) -> None:
        perq = {
            "q_sem_evidencia": {
                "col": 1,
                "llm": 1,
                "rag": 0,
                "rag_evidence": [],
                "rag_used_chunk_ids": [],
                "rag_reason": "no_explicit_evidence",
            },
            "q_com_evidencia": {
                "col": 0,
                "llm": 0,
                "rag": 1,
                "rag_evidence": ["Sistema Datasul declarado no documento."],
                "rag_used_chunk_ids": ["chunk-1"],
                "rag_reason": "",
            },
        }
        out = dm._compute_answered_from_per_question(perq, total_expected=2, rag_primary=True)
        self.assertEqual(out["answered"], 1)
        self.assertEqual(out["by_code"]["q_sem_evidencia"], 0)
        self.assertEqual(out["by_code"]["q_com_evidencia"], 1)
        self.assertEqual(perq["q_com_evidencia"]["answerable_source"], "rag")
        self.assertEqual(perq["q_com_evidencia"]["used_chunk_ids"], ["chunk-1"])
        self.assertTrue(perq["q_com_evidencia"]["evidence"])

    def test_compute_answered_uses_col_llm_when_not_rag_primary(self) -> None:
        perq = {
            "q_coluna": {
                "col": 1,
                "llm": 0,
                "rag": 0,
                "evidence_col": ["areas | linha 2 | campo: Sim"],
                "llm_evidence": [],
                "rag_evidence": [],
                "rag_used_chunk_ids": [],
                "rag_reason": "not_answerable",
            },
            "q_llm": {
                "col": 0,
                "llm": 1,
                "rag": 0,
                "evidence_col": [],
                "llm_evidence": ["Resposta direta no texto."],
                "rag_evidence": [],
                "rag_used_chunk_ids": [],
                "rag_reason": "not_answerable",
            },
        }
        out = dm._compute_answered_from_per_question(perq, total_expected=2, rag_primary=False)
        self.assertEqual(out["answered"], 2)
        self.assertEqual(out["by_code"]["q_coluna"], 1)
        self.assertEqual(out["by_code"]["q_llm"], 1)
        self.assertEqual(perq["q_coluna"]["answerable_source"], "col_llm")
        self.assertTrue(perq["q_coluna"]["evidence"])
        self.assertTrue(perq["q_llm"]["evidence"])

    def test_compute_answered_prioritizes_qa_block(self) -> None:
        perq = {
            "q_qa": {
                "qa": 1,
                "qa_evidence": ["Nome do sistema: Datasul"],
                "qa_used_chunk_ids": ["qa:sistema_nome:0"],
                "qa_reason": "",
                "col": 0,
                "llm": 0,
                "rag": 0,
                "rag_reason": "not_answerable",
            }
        }
        out = dm._compute_answered_from_per_question(perq, total_expected=1, rag_primary=True)
        self.assertEqual(out["answered"], 1)
        self.assertEqual(perq["q_qa"]["answerable_source"], "qa_block")
        self.assertEqual(perq["q_qa"]["used_chunk_ids"], ["qa:sistema_nome:0"])
        self.assertTrue(perq["q_qa"]["evidence"])

    def test_compute_answered_drops_prompt_only_evidence(self) -> None:
        perq = {
            "q_prompt": {
                "question": "Possui NDA?",
                "col": 0,
                "llm": 1,
                "rag": 0,
                "evidence_col": [],
                "llm_evidence": ["Possui NDA?"],
                "rag_evidence": [],
                "rag_used_chunk_ids": [],
                "rag_reason": "not_answerable",
            }
        }
        out = dm._compute_answered_from_per_question(perq, total_expected=1, rag_primary=False)
        self.assertEqual(out["answered"], 0)
        self.assertEqual(perq["q_prompt"]["answerable"], 0)
        self.assertEqual(perq["q_prompt"]["reason"], "filtered_evidence_empty")

    def test_compute_answered_drops_generic_evidence(self) -> None:
        perq = {
            "q_generic": {
                "question": "Onde fica armazenado?",
                "qa": 1,
                "qa_evidence": ["local de armazenamento dos dados"],
                "qa_used_chunk_ids": ["qa:hd_armazenado_onde:0"],
                "qa_reason": "",
                "col": 0,
                "llm": 0,
                "rag": 0,
            }
        }
        out = dm._compute_answered_from_per_question(perq, total_expected=1, rag_primary=True)
        self.assertEqual(out["answered"], 0)
        self.assertEqual(perq["q_generic"]["answerable"], 0)
        self.assertEqual(perq["q_generic"]["reason"], "filtered_evidence_empty")

    def test_compute_answered_drops_header_like_evidence(self) -> None:
        perq = {
            "q_header": {
                "question": "2- Possui tratamento de dados pessoais?",
                "col": 0,
                "llm": 1,
                "rag": 0,
                "evidence_col": [],
                "llm_evidence": ["Categoria dos dados (pessoal ou sensivel)"],
                "rag_evidence": [],
                "rag_used_chunk_ids": [],
                "rag_reason": "not_answerable",
            }
        }
        out = dm._compute_answered_from_per_question(perq, total_expected=1, rag_primary=False)
        self.assertEqual(out["answered"], 0)
        self.assertEqual(perq["q_header"]["answerable"], 0)
        self.assertEqual(perq["q_header"]["reason"], "filtered_evidence_empty")


if __name__ == "__main__":
    unittest.main()
