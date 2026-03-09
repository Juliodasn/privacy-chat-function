import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AvaliacaoDataMapping.answerability_guard import validate_answerability  # noqa: E402


class EvidenceAnchoringRegressionTests(unittest.TestCase):
    def test_regression_matrix(self) -> None:
        cases = [
            {
                "name": "descriptive_yes_no_direct",
                "question_code": "2_compart_dados_rede",
                "question_text": "Possui compartilhamento de dados na rede?",
                "question_keywords": {"required": ["rede"], "optional": ["compartilhamento"]},
                "evidence": ["Rede dedicada interna"],
                "chunk": "Rede dedicada interna",
                "expected": (1, "", ["Rede dedicada interna"]),
            },
            {
                "name": "prompt_only_question_drop",
                "question_code": "2_compart_dados_rede",
                "question_text": "Possui compartilhamento de dados na rede?",
                "question_keywords": {"required": ["rede"], "optional": ["compartilhamento"]},
                "evidence": ["Possui compartilhamento de dados na rede?"],
                "chunk": "Possui compartilhamento de dados na rede?",
                "expected": (0, "evidence_is_prompt", []),
            },
            {
                "name": "header_only_colon_drop",
                "question_code": "2_trata_dados_pessoais",
                "question_text": "Quais dados pessoais sao processados?",
                "question_keywords": {"required": ["cpf", "nome"], "optional": ["email"]},
                "evidence": ["Categoria dos dados:"],
                "chunk": "Categoria dos dados:",
                "expected": (0, "evidence_header_like", []),
            },
            {
                "name": "field_value_hyphen_pass",
                "question_code": "11_tempo_necessario_apenas",
                "question_text": "Qual o tempo de retencao?",
                "question_keywords": {"required": ["retencao"], "optional": ["permanente"]},
                "evidence": ["Retencao - Permanente"],
                "chunk": "Retencao - Permanente",
                "expected": (1, "", ["Retencao - Permanente"]),
            },
            {
                "name": "field_value_colon_yes_pass",
                "question_code": "4_1_contrato_apresentado",
                "question_text": "Contrato apresentado?",
                "question_keywords": {"required": ["contrato"], "optional": ["apresentado"]},
                "evidence": ["Contrato apresentado: Sim"],
                "chunk": "Contrato apresentado: Sim",
                "expected": (1, "", ["Contrato apresentado: Sim"]),
            },
            {
                "name": "expand_email",
                "question_code": "13_forma_transferencia_dados",
                "question_text": "Qual a forma de transferencia de dados?",
                "question_keywords": {"required": ["transferencia"], "optional": ["email"]},
                "evidence": ["E-mail"],
                "chunk": "Forma de transferencia: E-mail",
                "expected": (1, "auto_expanded_short_evidence", ["Forma de transferencia: E-mail"]),
            },
            {
                "name": "expand_permanente",
                "question_code": "11_tempo_necessario_apenas",
                "question_text": "Qual o tempo de retencao?",
                "question_keywords": {"required": ["retencao"], "optional": ["permanente"]},
                "evidence": ["Permanente"],
                "chunk": "Retencao - Permanente",
                "expected": (1, "auto_expanded_short_evidence", ["Retencao - Permanente"]),
            },
            {
                "name": "expand_sim_contract",
                "question_code": "4_1_contrato_apresentado",
                "question_text": "Contrato apresentado?",
                "question_keywords": {"required": ["contrato"], "optional": ["apresentado"]},
                "evidence": ["Sim"],
                "chunk": "Contrato apresentado: Sim",
                "expected": (1, "auto_expanded_short_evidence", ["Contrato apresentado: Sim"]),
            },
            {
                "name": "block_sim_substring_in_sistema",
                "question_code": "4_1_contrato_apresentado",
                "question_text": "Contrato apresentado?",
                "question_keywords": {"required": ["contrato"], "optional": ["apresentado"]},
                "evidence": ["Sim"],
                "chunk": "Sistema integrado principal",
                "expected": (0, "evidence_too_short", []),
            },
            {
                "name": "expand_nao_terceiros",
                "question_code": "4_compart_terceiros",
                "question_text": "Possui compartilhamento com terceiros?",
                "question_keywords": {"required": ["terceiros"], "optional": ["compartilhamento"]},
                "evidence": ["Nao"],
                "chunk": "Compartilhamento com terceiros: Nao",
                "expected": (1, "auto_expanded_short_evidence", ["Compartilhamento com terceiros: Nao"]),
            },
            {
                "name": "prompt_with_payload_yes_pass",
                "question_code": "4_2_termo_sigilo_terceiros",
                "question_text": "Possui NDA?",
                "question_keywords": {"required": ["nda"], "optional": ["sigilo"]},
                "evidence": ["Possui NDA? Sim"],
                "chunk": "Possui NDA? Sim",
                "expected": (1, "", ["Possui NDA? Sim"]),
            },
            {
                "name": "header_pipe_without_payload_drop",
                "question_code": "6_dados_tratados_quais",
                "question_text": "Quais dados sao tratados?",
                "question_keywords": {"required": ["dados"], "optional": ["tratados"]},
                "evidence": ["Dados tratados |"],
                "chunk": "Dados tratados |",
                "expected": (0, "evidence_header_like", []),
            },
            {
                "name": "repair_prompt_to_data_list",
                "question_code": "2_trata_dados_pessoais",
                "question_text": "Quais dados pessoais sao processados?",
                "question_keywords": {"required": ["cpf", "nome"], "optional": ["email"]},
                "evidence": ["Quais dados pessoais sao processados?"],
                "chunk": "Dados tratados: CPF, Nome, Email",
                "expected": (1, "auto_repaired_evidence_from_chunk", ["Dados tratados: CPF, Nome, Email"]),
            },
            {
                "name": "repair_prompt_to_location",
                "question_code": "1_2_hd_armazenado_onde",
                "question_text": "Onde os dados ficam armazenados?",
                "question_keywords": {"required": ["armazenamento"], "optional": ["servidor"]},
                "evidence": ["Onde os dados ficam armazenados?"],
                "chunk": "Local de armazenamento: Servidor interno",
                "expected": (1, "auto_repaired_evidence_from_chunk", ["Local de armazenamento: Servidor interno"]),
            },
            {
                "name": "prompt_only_chunk_rejected",
                "question_code": "4_2_termo_sigilo_terceiros",
                "question_text": "Possui NDA?",
                "question_keywords": {"required": ["nda"], "optional": ["sigilo"]},
                "evidence": ["Possui NDA?"],
                "chunk": "Possui NDA?",
                "expected": (0, "evidence_is_prompt", []),
            },
            {
                "name": "field_value_list_passes",
                "question_code": "2_trata_dados_pessoais",
                "question_text": "Quais dados pessoais sao processados?",
                "question_keywords": {"required": ["cpf", "nome"], "optional": ["email"]},
                "evidence": ["Dados tratados - CPF, Nome, Email"],
                "chunk": "Dados tratados - CPF, Nome, Email",
                "expected": (1, "", ["Dados tratados - CPF, Nome, Email"]),
            },
            {
                "name": "direct_location_value_passes",
                "question_code": "1_2_hd_armazenado_onde",
                "question_text": "Onde os dados ficam armazenados?",
                "question_keywords": {"required": ["armazenamento"], "optional": ["servidor"]},
                "evidence": ["Servidor interno"],
                "chunk": "Local de armazenamento: Servidor interno",
                "expected": (1, "", ["Servidor interno"]),
            },
            {
                "name": "expand_nda",
                "question_code": "4_2_termo_sigilo_terceiros",
                "question_text": "Possui NDA?",
                "question_keywords": {"required": ["nda"], "optional": ["sigilo"]},
                "evidence": ["NDA"],
                "chunk": "Termo NDA assinado",
                "expected": (1, "auto_repaired_evidence_from_chunk", ["Termo NDA assinado"]),
            },
            {
                "name": "expand_cpf",
                "question_code": "2_trata_dados_pessoais",
                "question_text": "Quais dados pessoais sao processados?",
                "question_keywords": {"required": ["cpf"], "optional": ["nome"]},
                "evidence": ["CPF"],
                "chunk": "Dados tratados: CPF, Nome",
                "expected": (1, "auto_repaired_evidence_from_chunk", ["Dados tratados: CPF, Nome"]),
            },
            {
                "name": "repair_sim_to_descriptive_yes_no",
                "question_code": "2_compart_dados_rede",
                "question_text": "Possui compartilhamento de dados na rede?",
                "question_keywords": {"required": ["rede"], "optional": ["compartilhamento"]},
                "evidence": ["Sim"],
                "chunk": "Rede dedicada interna",
                "expected": (1, "auto_repaired_evidence_from_chunk", ["Rede dedicada interna"]),
            },
            {
                "name": "field_value_supplier_not_header",
                "question_code": "2_fornecedor_identificado",
                "question_text": "Fornecedor identificado?",
                "question_keywords": {"required": ["fornecedor"], "optional": ["acme"]},
                "evidence": ["Fornecedor: ACME Ltda"],
                "chunk": "Fornecedor: ACME Ltda",
                "expected": (1, "", ["Fornecedor: ACME Ltda"]),
            },
            {
                "name": "bullet_list_passes",
                "question_code": "6_dados_tratados_quais",
                "question_text": "Quais dados sao tratados?",
                "question_keywords": {"required": ["cpf"], "optional": ["email"]},
                "evidence": ["- CPF, Nome, Email"],
                "chunk": "- CPF, Nome, Email",
                "expected": (1, "", ["- CPF, Nome, Email"]),
            },
            {
                "name": "missing_chunk_text",
                "question_code": "4_1_contrato_apresentado",
                "question_text": "Contrato apresentado?",
                "question_keywords": {"required": ["contrato"], "optional": ["apresentado"]},
                "evidence": ["Contrato apresentado: Sim"],
                "chunk": "",
                "expected": (0, "missing_chunk_text", ["Contrato apresentado: Sim"]),
            },
            {
                "name": "generic_drop",
                "question_code": "5_descricao_processo",
                "question_text": "Qual a descricao do processo?",
                "question_keywords": {"required": ["descricao"], "optional": ["processo"]},
                "evidence": ["Informacoes do processo"],
                "chunk": "Informacoes do processo",
                "expected": (0, "evidence_too_generic", []),
            },
            {
                "name": "paraphrased_wrong_chunk_rejected",
                "question_code": "4_1_contrato_apresentado",
                "question_text": "Contrato apresentado?",
                "question_keywords": {"required": ["contrato"], "optional": ["fornecedor"]},
                "evidence": ["Contrato assinado pelo fornecedor"],
                "chunk": "Sistema principal em producao",
                "expected": (0, "evidence_missing_in_chunk", []),
            },
        ]

        self.assertEqual(len(cases), 25)

        for case in cases:
            with self.subTest(case=case["name"]):
                out = validate_answerability(
                    question_code=case["question_code"],
                    answerable=1,
                    evidence=case["evidence"],
                    used_ids=["chunk-a"],
                    chunk_texts={"chunk-a": case["chunk"]},
                    question_text=case["question_text"],
                    question_keywords=case["question_keywords"],
                    strict_mode=True,
                )
                self.assertEqual(out, case["expected"])


if __name__ == "__main__":
    unittest.main()
