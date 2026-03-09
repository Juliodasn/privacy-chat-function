import sys
import unittest
from pathlib import Path
from AvaliacaoDataMapping.anchors_catalog import QUESTION_ANCHOR_MAP


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AvaliacaoDataMapping.qa_blocks import (  # noqa: E402
    extract_qa_blocks,
    normalize_na_variants,
    resolve_question_from_qa_blocks,
)


class QaBlocksTests(unittest.TestCase):
    def test_extract_qa_blocks_single_line(self) -> None:
        text = "Possui NDA (Clausula ou Contrato de Confidencialidade)?: n/a"
        qa_blocks, qa_index = extract_qa_blocks(text)
        self.assertIn("possui_nda", qa_index)
        idx = qa_index["possui_nda"][0]
        self.assertEqual(qa_blocks[idx]["answer_text"], "n/a")

    def test_extract_qa_blocks_multi_line_stop_on_next_anchor(self) -> None:
        text = "\n".join(
            [
                "Nome do sistema",
                "Datasul",
                "Fornecedor do sistema: TOTVS",
            ]
        )
        qa_blocks, qa_index = extract_qa_blocks(text)
        self.assertIn("sistema_nome", qa_index)
        idx = qa_index["sistema_nome"][0]
        block = qa_blocks[idx]
        self.assertIn("Datasul", block["answer_text"])
        self.assertEqual(block["span"]["line_end"], 2)

    def test_normalize_na_variants(self) -> None:
        self.assertEqual(normalize_na_variants("Nao se aplica"), "n/a")
        self.assertEqual(normalize_na_variants("N/A"), "n/a")
        self.assertEqual(normalize_na_variants("nao informado"), "n/a")

    def test_answerable_from_qa_block_has_evidence_and_used_id(self) -> None:
        text = "\n".join(
            [
                "Nome do sistema",
                "Datasul",
            ]
        )
        qa_blocks, qa_index = extract_qa_blocks(text)
        res = resolve_question_from_qa_blocks("1_nome_identificado", qa_blocks, qa_index)
        self.assertEqual(res["answerable"], 1)
        self.assertTrue(res["evidence"])
        self.assertTrue(res["used_chunk_ids"])
        self.assertTrue(res["used_chunk_ids"][0].startswith("qa:sistema_nome:"))

    def test_qa_rejects_answer_equal_to_anchor_header(self) -> None:
        qa_blocks = [
            {
                "anchor_id": "hd_armazenado_onde",
                "anchor_text": "Onde fica armazenado?",
                "answer_text": "Onde fica armazenado?",
                "confidence": 0.95,
                "source": "pdf_raw_text",
                "span": {"line_start": 1, "line_end": 1},
            }
        ]
        qa_index = {"hd_armazenado_onde": [0]}
        res = resolve_question_from_qa_blocks("1_2_hd_armazenado_onde", qa_blocks, qa_index)
        self.assertEqual(res["answerable"], 0)

    def test_qa_rejects_generic_location_answer(self) -> None:
        qa_blocks = [
            {
                "anchor_id": "hd_armazenado_onde",
                "anchor_text": "Onde fica armazenado?",
                "answer_text": "local de armazenamento dos dados",
                "confidence": 0.95,
                "source": "pdf_raw_text",
                "span": {"line_start": 1, "line_end": 1},
            }
        ]
        qa_index = {"hd_armazenado_onde": [0]}
        res = resolve_question_from_qa_blocks("1_2_hd_armazenado_onde", qa_blocks, qa_index)
        self.assertEqual(res["answerable"], 0)

    def test_extract_supplier_from_explicit_label(self) -> None:
        text = "Fornecedor do sistema: TOTVS"
        qa_blocks, qa_index = extract_qa_blocks(text)

        self.assertIn("sistema_fornecedor", qa_index)
        idx = qa_index["sistema_fornecedor"][0]
        self.assertEqual(qa_blocks[idx]["answer_text"], "TOTVS")

    def test_gap_provider_line_does_not_generate_system_supplier_qa(self) -> None:
        text = "\n".join(
            [
                "GAP 3: Identificar provedor de e-mail.",
                "GAP 4: Verificar se há transferência internacional de dados no provedor de e-mail",
            ]
        )

        qa_blocks, qa_index = extract_qa_blocks(text)

        self.assertNotIn("sistema_fornecedor", qa_index)

    def test_question_12_has_anchor_mapping_and_accepts_explicit_format_block(self) -> None:
        self.assertIn("12_formato_dados_disponiveis", QUESTION_ANCHOR_MAP)

        text = "Formato dos dados disponiveis: planilha e e-mail"
        qa_blocks, qa_index = extract_qa_blocks(text)

        self.assertIn("formato_dados_disponiveis", qa_index)

        res = resolve_question_from_qa_blocks("12_formato_dados_disponiveis", qa_blocks, qa_index)
        self.assertEqual(res["answerable"], 1)
        self.assertTrue(any("planilha" in ev.lower() for ev in res["evidence"]))

if __name__ == "__main__":
    unittest.main()
