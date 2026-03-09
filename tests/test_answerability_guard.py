import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AvaliacaoDataMapping.answerability_guard import validate_answerability  # noqa: E402
from AvaliacaoDataMapping.rag_engine import _sanitize_rag_json_response  # noqa: E402


class AnswerabilityGuardTests(unittest.TestCase):
    def test_yes_no_accepts_descriptive_value_like_evidence(self) -> None:
        out = validate_answerability(
            question_code="2_compart_dados_rede",
            answerable=1,
            evidence=["Rede pessoal interna da empresa"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Rede pessoal interna da empresa"},
            question_text="Possui compartilhamento de dados na rede?",
            strict_mode=True,
        )
        self.assertEqual(out, (1, "", ["Rede pessoal interna da empresa"]))

    def test_yes_no_rejects_prompt_only_evidence(self) -> None:
        out = validate_answerability(
            question_code="2_compart_dados_rede",
            answerable=1,
            evidence=["Possui compartilhamento de dados na rede?"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Possui compartilhamento de dados na rede?"},
            question_text="Possui compartilhamento de dados na rede?",
            strict_mode=True,
        )
        self.assertEqual(out, (0, "evidence_is_prompt", []))

    def test_expand_short_evidence_from_chunk(self) -> None:
        out = validate_answerability(
            question_code="13_forma_transferencia_dados",
            answerable=1,
            evidence=["E-mail"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Forma de transferencia: E-mail"},
            question_text="13. Qual a forma de transferencia de dados?",
            strict_mode=True,
        )
        self.assertEqual(
            out,
            (1, "auto_expanded_short_evidence", ["Forma de transferencia: E-mail"]),
        )

    def test_explicit_questions_require_anchor_terms(self) -> None:
        out = validate_answerability(
            question_code="4_2_termo_sigilo_terceiros",
            answerable=1,
            evidence=["O contexto menciona explicitamente o tratamento de dados pessoais em conformidade com a LGPD."],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "O contexto menciona explicitamente o tratamento de dados pessoais em conformidade com a LGPD."},
            question_text="4.2 Existe termo de sigilo?",
            strict_mode=True,
        )
        self.assertEqual(out, (0, "missing_explicit_anchor_for_question", []))

    def test_explicit_questions_accept_anchored_na_or_binary_value(self) -> None:
        out = validate_answerability(
            question_code="4_2_termo_sigilo_terceiros",
            answerable=1,
            evidence=["Possui NDA (Clausula ou Contrato de Confidencialidade)?: n/a"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Possui NDA (Clausula ou Contrato de Confidencialidade)?: n/a"},
            question_text="4.2 Existe termo de sigilo?",
            strict_mode=True,
        )
        self.assertEqual(out, (1, "", ["Possui NDA (Clausula ou Contrato de Confidencialidade)?: n/a"]))

    def test_repair_prompt_evidence_from_chunk(self) -> None:
        out = validate_answerability(
            question_code="2_trata_dados_pessoais",
            answerable=1,
            evidence=["Quais dados pessoais sao processados?"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Dados tratados: CPF, Nome, Email"},
            question_text="Quais dados pessoais sao processados?",
            strict_mode=True,
        )
        self.assertEqual(
            out,
            (1, "auto_repaired_evidence_from_chunk", ["Dados tratados: CPF, Nome, Email"]),
        )

    def test_sanitize_prefers_value_like_line_from_chunk(self) -> None:
        out = _sanitize_rag_json_response(
            {"answerable": 1, "evidence": [], "used_chunk_ids": ["chunk-a"]},
            {"chunk-a"},
            fallback_chunks=[
                {
                    "chunk_id": "chunk-a",
                    "text": (
                        "Possui compartilhamento de dados na rede?\n"
                        "Rede pessoal interna da empresa\n"
                        "Observacao adicional."
                    ),
                }
            ],
            question_code="2_compart_dados_rede",
            question_text="Possui compartilhamento de dados na rede?",
        )
        self.assertEqual(out["answerable"], 1)
        self.assertEqual(out["evidence"], ["Rede pessoal interna da empresa"])
        self.assertEqual(out["reason"], "")

    def test_sanitize_recovers_value_like_line_when_model_returns_prompt(self) -> None:
        out = _sanitize_rag_json_response(
            {
                "answerable": 1,
                "evidence": ["Possui compartilhamento de dados na rede?"],
                "used_chunk_ids": ["chunk-a"],
            },
            {"chunk-a"},
            fallback_chunks=[
                {
                    "chunk_id": "chunk-a",
                    "text": (
                        "Possui compartilhamento de dados na rede?\n"
                        "Rede pessoal interna da empresa"
                    ),
                }
            ],
            question_code="2_compart_dados_rede",
            question_text="Possui compartilhamento de dados na rede?",
        )
        self.assertEqual(out["answerable"], 1)
        self.assertEqual(out["evidence"], ["Rede pessoal interna da empresa"])
        self.assertEqual(out["reason"], "auto_repaired_evidence_from_chunk")

    def test_global_na_value_is_respondable_when_anchored(self) -> None:
        out = validate_answerability(
            question_code="6_3_compart_interno_terceiros",
            answerable=1,
            evidence=["Compartilhamento interno ou com terceiros: n/a"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Compartilhamento interno ou com terceiros: n/a"},
            question_text="6.3 Pode ser identificado se ha compartilhamento interno ou com terceiros?",
            strict_mode=True,
        )
        self.assertEqual(out, (1, "", ["Compartilhamento interno ou com terceiros: n/a"]))

    def test_sensitive_question_accepts_categoria_pessoal(self) -> None:
        out = validate_answerability(
            question_code="3_trata_dados_sensiveis",
            answerable=1,
            evidence=["Categoria dos dados (pessoal ou sensivel): Pessoal"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Categoria dos dados (pessoal ou sensivel): Pessoal"},
            question_text="3- Possui tratamento de dados sensiveis?",
            strict_mode=True,
        )
        self.assertEqual(out, (1, "", ["Categoria dos dados (pessoal ou sensivel): Pessoal"]))

    def test_access_restriction_question_accepts_role_based_access_evidence(self) -> None:
        out = validate_answerability(
            question_code="2_2_politica_restricao_acesso",
            answerable=1,
            evidence=["Gestao de acesso acontece por area + liderancas + controller + financeiro"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Gestao de acesso acontece por area + liderancas + controller + financeiro"},
            question_text="2.2 Existe politica de restricao de acesso?",
            strict_mode=True,
        )
        self.assertEqual(
            out,
            (1, "", ["Gestao de acesso acontece por area + liderancas + controller + financeiro"]),
        )

    def test_contract_header_with_sim_is_respondable(self) -> None:
        out = validate_answerability(
            question_code="4_possui_contrato",
            answerable=1,
            evidence=["Possui contrato escrito assinado?: sim"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Possui contrato escrito assinado?: sim"},
            question_text="4- E possivel identificar se este sistema possui algum contrato com o fornecedor?",
            strict_mode=True,
        )
        self.assertEqual(out, (1, "", ["Possui contrato escrito assinado?: sim"]))

    def test_format_question_accepts_planilha_e_email_as_explicit_format(self) -> None:
        out = validate_answerability(
            question_code="12_formato_dados_disponiveis",
            answerable=1,
            evidence=["Planilha enviada por e-mail"],
            used_ids=["chunk-a"],
            chunk_texts={"chunk-a": "Planilha enviada por e-mail"},
            question_text="12- Esta documentado o formato no qual os dados estao disponiveis?",
            strict_mode=True,
        )
        self.assertEqual(out, (1, "", ["Planilha enviada por e-mail"]))


if __name__ == "__main__":
    unittest.main()
