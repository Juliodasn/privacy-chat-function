import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AvaliacaoDataMapping.rag_engine import (  # noqa: E402
    _default_question_keywords,
    _split_free_text_blocks,
    _sanitize_rag_json_response,
    RagChunk,
    build_chunks,
    evaluate_questions_with_rag,
    retrieve_context_for_question,
)
from AvaliacaoDataMapping.anchors_catalog import QUESTION_KEYWORDS  # noqa: E402


class RagEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prev_openai = os.environ.get("OPENAI_API_KEY")
        self.prev_azure_a = os.environ.get("AZURE_OPENAI_API_KEY")
        self.prev_azure_b = os.environ.get("AZURE_OPENAI_KEY")
        self.prev_strict = os.environ.get("STRICT_MODE")
        os.environ["OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_API_KEY"] = ""
        os.environ["AZURE_OPENAI_KEY"] = ""
        os.environ["STRICT_MODE"] = "true"

    def tearDown(self) -> None:
        def _restore(key: str, value: str | None) -> None:
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

        _restore("OPENAI_API_KEY", self.prev_openai)
        _restore("AZURE_OPENAI_API_KEY", self.prev_azure_a)
        _restore("AZURE_OPENAI_KEY", self.prev_azure_b)
        _restore("STRICT_MODE", self.prev_strict)

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

    def test_lexical_rerank_promotes_correct_chunk(self) -> None:
        index = {
            "has_embeddings": True,
            "chunks": [
                RagChunk("c_wrong", "sistemas", "text:block", "Sistema legado sem identificacao clara.", {}),
                RagChunk("c_right", "sistemas", "text:block", "Nome do sistema: Datasul", {}),
            ],
            "embeddings": [[1.0, 0.0], [0.97, 0.0]],
            "section_positions": {"sistemas": [0, 1]},
            "question_sections": {"q_sys": "sistemas"},
            "question_keywords": {
                "q_sys": {
                    "required": ["datasul"],
                    "optional": ["nome", "sistema"],
                    "negative": [],
                }
            },
            "_question_embedding_cache": {"q_sys::nome do sistema": [1.0, 0.0]},
            "_embedding_client": None,
            "embedding_model": None,
        }
        out = retrieve_context_for_question("q_sys", "nome do sistema", index, top_k=1, include_debug=True)
        self.assertTrue(isinstance(out, dict))
        self.assertEqual(out["debug"]["before_rerank"][0]["chunk_id"], "c_wrong")
        self.assertEqual(out["debug"]["after_rerank"][0]["chunk_id"], "c_right")
        self.assertEqual(out["hits"][0]["chunk_id"], "c_right")

    def test_retrieval_pool_combines_embedding_and_lexical_candidates(self) -> None:
        prev_alpha = os.environ.get("RAG_RETRIEVAL_ALPHA")
        try:
            os.environ["RAG_RETRIEVAL_ALPHA"] = "0.2"
            index = {
                "has_embeddings": True,
                "chunks": [
                    RagChunk("c1", "processos", "text:block", "Texto generico sem termos relevantes.", {}),
                    RagChunk("c2", "processos", "text:block", "Texto generico sem termos relevantes.", {}),
                    RagChunk("c3", "processos", "text:block", "Texto generico sem termos relevantes.", {}),
                    RagChunk("c4", "processos", "text:block", "Texto generico sem termos relevantes.", {}),
                    RagChunk("c5", "processos", "text:block", "Gestao de acesso por area e controle de perfil.", {}),
                ],
                "embeddings": [[1.0, 0.0], [0.99, 0.0], [0.98, 0.0], [0.97, 0.0], [0.0, 1.0]],
                "section_positions": {"processos": [0, 1, 2, 3, 4]},
                "question_sections": {"q_sec": "processos"},
                "question_keywords": {
                    "q_sec": {
                        "required": ["acesso"],
                        "optional": ["gestao", "controle", "perfil"],
                        "negative": [],
                    }
                },
                "_question_embedding_cache": {"q_sec::medidas de acesso": [1.0, 0.0]},
                "_embedding_client": None,
                "embedding_model": None,
            }
            out = retrieve_context_for_question("q_sec", "medidas de acesso", index, top_k=1, include_debug=True)
            self.assertEqual(out["hits"][0]["chunk_id"], "c5")
            before_ids = {row["chunk_id"] for row in out["debug"]["before_rerank"]}
            self.assertIn("c5", before_ids)
        finally:
            if prev_alpha is None:
                os.environ.pop("RAG_RETRIEVAL_ALPHA", None)
            else:
                os.environ["RAG_RETRIEVAL_ALPHA"] = prev_alpha

    def test_split_free_text_blocks_collapses_pdfminer_micro_paragraphs(self) -> None:
        text = "\n\n".join(f"Linha {i}" for i in range(1, 31))
        blocks = _split_free_text_blocks(text)
        self.assertLessEqual(len(blocks), 6)
        self.assertTrue(any("\n" in block for block in blocks))

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
        self.assertEqual(out["meta"]["guard_dropped_answerable_total"], 0)
        self.assertEqual(out["meta"]["guard_auto_repair_total"], 0)
        self.assertEqual(out["meta"]["guard_auto_expand_total"], 0)
        self.assertEqual(out["results"]["q1"]["reason"], "no_chat_client")
        self.assertEqual(out["results"]["q1"]["rag_chunks"][0]["chunk_id"], "a1")

    def test_evaluate_questions_with_rag_prefers_schema_table_network_evidence(self) -> None:
        index = {
            "has_embeddings": True,
            "chunks": [
                RagChunk(
                    "a1",
                    "areas",
                    "table:areas",
                    "Compartilhamento de dados na rede: Rede pessoal interna da empresa",
                    {},
                )
            ],
            "embeddings": [[1.0, 0.0]],
            "section_positions": {"areas": [0]},
            "question_sections": {"2_compart_dados_rede": "areas"},
            "_question_embedding_cache": {"2_compart_dados_rede::Possui compartilhamento de dados na rede?": [1.0, 0.0]},
            "_embedding_client": None,
            "embedding_model": None,
        }
        out = evaluate_questions_with_rag(
            [("2_compart_dados_rede", "Possui compartilhamento de dados na rede?")],
            index,
            top_k=2,
            batch_size=2,
            max_context_chars=4000,
            max_context_tokens=1000,
        )
        row = out["results"]["2_compart_dados_rede"]
        self.assertEqual(row["answerable"], 1)
        self.assertEqual(row["reason"], "schema_table_value_present")
        self.assertTrue(any("rede pessoal interna da empresa" in ev.lower() for ev in row["evidence"]))
        self.assertEqual(row["used_chunk_ids"], ["a1"])
        self.assertEqual(out["meta"]["schema_table_applied_total"], 1)

    def test_evaluate_questions_with_rag_schema_na_is_respondable_without_chat_client(self) -> None:
        index = {
            "has_embeddings": True,
            "chunks": [
                RagChunk(
                    "a1",
                    "areas",
                    "table:areas",
                    "Possui NDA (Clausula ou Contrato de Confidencialidade)?: n/a",
                    {},
                )
            ],
            "embeddings": [[1.0, 0.0]],
            "section_positions": {"areas": [0]},
            "question_sections": {"4_2_termo_sigilo_terceiros": "areas"},
            "_question_embedding_cache": {"4_2_termo_sigilo_terceiros::Existe termo de sigilo?": [1.0, 0.0]},
            "_embedding_client": None,
            "embedding_model": None,
        }
        out = evaluate_questions_with_rag(
            [("4_2_termo_sigilo_terceiros", "Existe termo de sigilo?")],
            index,
            top_k=2,
            batch_size=2,
            max_context_chars=4000,
            max_context_tokens=1000,
        )
        row = out["results"]["4_2_termo_sigilo_terceiros"]
        self.assertEqual(row["answerable"], 1)
        self.assertEqual(row["reason"], "schema_value_present_na")
        self.assertTrue(any("n/a" in ev.lower() for ev in row["evidence"]))
        self.assertEqual(row["used_chunk_ids"], ["a1"])
        self.assertEqual(out["meta"]["schema_table_applied_total"], 1)

    def test_sanitize_strict_mode_blocks_auto_assign_chunk(self) -> None:
        out = _sanitize_rag_json_response(
            {"answerable": 1, "evidence": [], "used_chunk_ids": [], "reason": ""},
            {"chunk-top-1"},
            fallback_chunks=[{"chunk_id": "chunk-top-1", "text": "Sistema Datasul para operacao principal."}],
        )
        self.assertEqual(out["answerable"], 0)
        self.assertEqual(out["reason"], "strict_missing_chunk_trace")

    def test_sanitize_fills_missing_evidence_from_retrieved_chunk(self) -> None:
        out = _sanitize_rag_json_response(
            {"answerable": 1, "evidence": [], "used_chunk_ids": ["chunk-a"]},
            {"chunk-a"},
            fallback_chunks=[{"chunk_id": "chunk-a", "text": "Enviar copia do contrato assinado ao fornecedor."}],
        )
        self.assertEqual(out["answerable"], 1)
        self.assertEqual(out["used_chunk_ids"], ["chunk-a"])
        self.assertTrue(out["evidence"])

    def test_sanitize_legacy_mode_allows_auto_assign_chunk(self) -> None:
        os.environ["STRICT_MODE"] = "false"
        out = _sanitize_rag_json_response(
            {"answerable": 1, "evidence": [], "used_chunk_ids": []},
            {"chunk-top-1"},
            fallback_chunks=[{"chunk_id": "chunk-top-1", "text": "Sistema Datasul para operacao principal."}],
        )
        self.assertEqual(out["answerable"], 1)
        self.assertEqual(out["used_chunk_ids"], ["chunk-top-1"])
        self.assertEqual(out["reason"], "auto_assigned_chunk")

    def test_evaluate_questions_with_rag_detects_access_restriction_policy_from_role_based_access(self) -> None:
        index = {
            "has_embeddings": True,
            "chunks": [
                RagChunk(
                    "a1",
                    "areas",
                    "table:areas",
                    "Quem pode acessar os dados?: Gestores\nGestao de acesso acontece por area + liderancas + controller + financeiro",
                    {},
                )
            ],
            "embeddings": [[1.0, 0.0]],
            "section_positions": {"areas": [0]},
            "question_sections": {"2_2_politica_restricao_acesso": "areas"},
            "_question_embedding_cache": {
                "2_2_politica_restricao_acesso::Existe politica de restricao de acesso?": [1.0, 0.0]
            },
            "_embedding_client": None,
            "embedding_model": None,
        }
        out = evaluate_questions_with_rag(
            [("2_2_politica_restricao_acesso", "Existe politica de restricao de acesso?")],
            index,
            top_k=2,
            batch_size=2,
            max_context_chars=4000,
            max_context_tokens=1000,
        )
        row = out["results"]["2_2_politica_restricao_acesso"]
        self.assertEqual(row["answerable"], 1)
        self.assertEqual(row["reason"], "schema_table_value_present")
        self.assertTrue(any("gestao de acesso" in ev.lower() for ev in row["evidence"]))
        self.assertEqual(row["used_chunk_ids"], ["a1"])

    def test_evaluate_questions_with_rag_does_not_mark_necessario_from_generic_purpose_text(self) -> None:
        index = {
            "has_embeddings": True,
            "chunks": [
                RagChunk(
                    "p1",
                    "processos",
                    "table:processos",
                    (
                        "Por que estes dados sao processados?\n"
                        "CPF - Esocial ; Data Nascimento - Idade ; Endereco - Distancia, rotas, consultas internas"
                    ),
                    {},
                )
            ],
            "embeddings": [[1.0, 0.0]],
            "section_positions": {"processos": [0]},
            "question_sections": {"9_somente_quando_necessario": "processos"},
            "question_keywords": {"9_somente_quando_necessario": QUESTION_KEYWORDS["9_somente_quando_necessario"]},
            "_question_embedding_cache": {
                "9_somente_quando_necessario::9- Os dados sao utilizados somente quando necessario?": [1.0, 0.0]
            },
            "_embedding_client": None,
            "embedding_model": None,
        }
        out = evaluate_questions_with_rag(
            [("9_somente_quando_necessario", "9- Os dados sao utilizados somente quando necessario?")],
            index,
            top_k=2,
            batch_size=2,
            max_context_chars=4000,
            max_context_tokens=1000,
        )
        row = out["results"]["9_somente_quando_necessario"]
        self.assertEqual(row["answerable"], 0)
        self.assertEqual(row["reason"], "no_chat_client")
        self.assertEqual(out["meta"]["schema_table_applied_total"], 0)

    def test_evaluate_questions_with_rag_accepts_explicit_necessario_anchor(self) -> None:
        index = {
            "has_embeddings": True,
            "chunks": [
                RagChunk(
                    "p1",
                    "processos",
                    "table:processos",
                    "Os dados pessoais coletados sao estritamente necessarios para que o processo de negocio atinja o seu objetivo: Sim",
                    {},
                )
            ],
            "embeddings": [[1.0, 0.0]],
            "section_positions": {"processos": [0]},
            "question_sections": {"9_somente_quando_necessario": "processos"},
            "question_keywords": {"9_somente_quando_necessario": QUESTION_KEYWORDS["9_somente_quando_necessario"]},
            "_question_embedding_cache": {
                "9_somente_quando_necessario::9- Os dados sao utilizados somente quando necessario?": [1.0, 0.0]
            },
            "_embedding_client": None,
            "embedding_model": None,
        }
        out = evaluate_questions_with_rag(
            [("9_somente_quando_necessario", "9- Os dados sao utilizados somente quando necessario?")],
            index,
            top_k=2,
            batch_size=2,
            max_context_chars=4000,
            max_context_tokens=1000,
        )
        row = out["results"]["9_somente_quando_necessario"]
        self.assertEqual(row["answerable"], 1)
        self.assertEqual(row["reason"], "schema_table_value_present")
        self.assertTrue(any("estritamente necessarios" in ev.lower() for ev in row["evidence"]))
        self.assertEqual(row["used_chunk_ids"], ["p1"])

    def test_sanitize_rejects_prompt_evidence(self) -> None:
        out = _sanitize_rag_json_response(
            {"answerable": 1, "evidence": ["Possui NDA?"], "used_chunk_ids": ["chunk-a"]},
            {"chunk-a"},
            fallback_chunks=[{"chunk_id": "chunk-a", "text": "Possui NDA?"}],
            question_code="4_2_termo_sigilo_terceiros",
            question_text="4.2 Existe termo de sigilo?",
        )
        self.assertEqual(out["answerable"], 0)
        self.assertEqual(out["reason"], "evidence_is_prompt")

    def test_sanitize_rejects_evidence_missing_in_used_chunk(self) -> None:
        out = _sanitize_rag_json_response(
            {"answerable": 1, "evidence": ["Enviar copia do contrato"], "used_chunk_ids": ["chunk-a"]},
            {"chunk-a"},
            fallback_chunks=[{"chunk_id": "chunk-a", "text": "Texto sem correspondencia literal."}],
            question_code="4_1_contrato_apresentado",
            question_text="Contrato apresentado?",
        )
        self.assertEqual(out["answerable"], 0)
        self.assertEqual(out["reason"], "evidence_missing_in_chunk")

    def test_sanitize_rejects_header_only_evidence(self) -> None:
        out = _sanitize_rag_json_response(
            {"answerable": 1, "evidence": ["Categoria dos dados (pessoal ou sensivel)"], "used_chunk_ids": ["chunk-a"]},
            {"chunk-a"},
            fallback_chunks=[{"chunk_id": "chunk-a", "text": "Categoria dos dados (pessoal ou sensivel)"}],
            question_code="2_trata_dados_pessoais",
            question_text="2- Possui tratamento de dados pessoais?",
        )
        self.assertEqual(out["answerable"], 0)
        self.assertEqual(out["reason"], "evidence_header_like")

    def test_sanitize_accepts_sensitive_category_pessoal(self) -> None:
        out = _sanitize_rag_json_response(
            {
                "answerable": 1,
                "evidence": ["Categoria dos dados (pessoal ou sensivel): Pessoal"],
                "used_chunk_ids": ["chunk-a"],
            },
            {"chunk-a"},
            fallback_chunks=[{"chunk_id": "chunk-a", "text": "Categoria dos dados (pessoal ou sensivel): Pessoal"}],
            question_code="3_trata_dados_sensiveis",
            question_text="3- Possui tratamento de dados sensiveis?",
        )
        self.assertEqual(out["answerable"], 1)
        self.assertTrue(out["evidence"])

    def test_sanitize_accepts_contract_cell_with_sim(self) -> None:
        out = _sanitize_rag_json_response(
            {
                "answerable": 1,
                "evidence": ["Possui contrato escrito assinado?: sim"],
                "used_chunk_ids": ["chunk-a"],
            },
            {"chunk-a"},
            fallback_chunks=[{"chunk_id": "chunk-a", "text": "Possui contrato escrito assinado?: sim"}],
            question_code="4_possui_contrato",
            question_text="4- E possivel identificar se este sistema possui algum contrato com o fornecedor?",
        )
        self.assertEqual(out["answerable"], 1)
        self.assertTrue(out["evidence"])

    def test_sanitize_accepts_personal_data_terms_without_sim_nao(self) -> None:
        out = _sanitize_rag_json_response(
            {
                "answerable": 1,
                "evidence": ["Dados tratados: CPF, nome e email"],
                "used_chunk_ids": ["chunk-a"],
            },
            {"chunk-a"},
            fallback_chunks=[{"chunk_id": "chunk-a", "text": "Dados tratados: CPF, nome e email"}],
            question_code="2_trata_dados_pessoais",
            question_text="2- Possui tratamento de dados pessoais?",
        )
        self.assertEqual(out["answerable"], 1)
        self.assertTrue(out["evidence"])

    def test_sanitize_expands_short_evidence(self) -> None:
        out = _sanitize_rag_json_response(
            {
                "answerable": 1,
                "evidence": ["E-mail"],
                "used_chunk_ids": ["chunk-a"],
            },
            {"chunk-a"},
            fallback_chunks=[{"chunk_id": "chunk-a", "text": "Forma de transferencia: E-mail"}],
            question_code="13_forma_transferencia_dados",
            question_text="13. Qual a forma de transferencia de dados?",
        )
        self.assertEqual(out["answerable"], 1)
        self.assertEqual(out["evidence"], ["Forma de transferencia: E-mail"])
        self.assertEqual(out["reason"], "auto_expanded_short_evidence")

    def test_sanitize_exposes_guard_debug_via_debug_out(self) -> None:
        debug = {}
        out = _sanitize_rag_json_response(
            {
                "answerable": 1,
                "evidence": ["Contrato apresentado: Sim"],
                "used_chunk_ids": ["chunk-a"],
            },
            {"chunk-a"},
            fallback_chunks=[{"chunk_id": "chunk-a", "text": "Contrato apresentado: Sim"}],
            question_code="4_1_contrato_apresentado",
            question_text="Contrato apresentado?",
            question_keywords={"required": ["contrato"], "optional": ["apresentado"]},
            debug_out=debug,
        )
        self.assertEqual(out["answerable"], 1)
        self.assertIn("guard_anchoring_score", debug)
        self.assertIn("guard_selected_fragment_type", debug)
        self.assertIn("guard_keyword_hits_required", debug)
        self.assertIn("guard_keyword_hits_optional", debug)

    def test_default_question_keywords_filters_generic_terms(self) -> None:
        out = _default_question_keywords("Existe contrato com terceiros e NDA?")
        self.assertIn("contrato", out["required"] + out["optional"])
        self.assertIn("terceiros", out["required"] + out["optional"])
        self.assertNotIn("existe", out["required"] + out["optional"])

    def test_question_keywords_include_security_hints(self) -> None:
        self.assertIn("17_medidas_seg_cibernetica", QUESTION_KEYWORDS)
        self.assertIn("17_2_medidas_seg_descritas", QUESTION_KEYWORDS)
        self.assertIn("acesso", QUESTION_KEYWORDS["17_medidas_seg_cibernetica"]["required"])
        self.assertIn("controle", QUESTION_KEYWORDS["17_2_medidas_seg_descritas"]["optional"])


if __name__ == "__main__":
    unittest.main()
