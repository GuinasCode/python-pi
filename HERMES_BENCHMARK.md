# Fase 19 — Benchmark contra Hermes (Nous Research)

> plan.md, seção 23/29: usar o Hermes Agent como benchmark de capacidades, não como alvo de
> clonagem. Classificar cada capacidade como `not implemented` / `partial` / `equivalent` /
> `better`, sempre com evidência. **Nunca declarar "melhor" sem evidência.**

## Metodologia e limitação honesta

Esta sessão não teve acesso a uma instância real, executável, do Hermes Agent (sem acesso à
rede/repositório do Hermes neste ambiente). Isso significa que **nenhuma badge `equivalent` ou
`better` pode ser atribuída com evidência real** — fazê-lo violaria a própria regra da Fase 19.

O que este documento oferece em vez disso, honestamente:

1. Um inventário real e verificável do que o `python-pi` (branch `research-first-runtime`,
   Fases 0-18) efetivamente implementa, com evidência `arquivo:linha` e testes que comprovam
   cada afirmação.
2. Uma classificação **relativa às capacidades publicamente conhecidas de um Hermes-like
   research agent** (planejamento, pesquisa multi-step, delegação, memória, skills,
   verificação) — sem comparação linha a linha contra código do Hermes que não foi lido nesta
   sessão.
3. Onde a capacidade do Hermes não pôde ser verificada, a badge é `not implemented` do lado da
   *comparação* (não do nosso lado), com a ressalva explícita.

## Matriz de capacidades

| Capacidade | Nosso status | Evidência | Badge |
|---|---|---|---|
| Coding (edit/read/write/bash/grep) | Real, builtin, testado | `pi_coding_agent/tools.py`; `tests/pi_coding_agent/test_tools.py` | partial — comparação não verificada |
| Web research (evidence-first) | Real, mas escopo mínimo: extração via URL fornecida, sem search provider real | `src/pi_runtime/research.py`; `tests/pi_runtime/test_research.py` (12 testes) | partial |
| Browser automation | Real via Playwright, sessão one-shot (sem página persistente interativa) | `src/pi_runtime/browser.py`; `tests/pi_runtime/test_browser.py` (15 testes) | partial |
| Memory (cognitiva) | Real: SQLite+FTS5+sqlite-vec, taxonomia Working/Episodic/Semantic/Procedural/User/Project, dedupe, secret guard | `src/pi_memory/store.py`, `src/pi_runtime/memory.py`; `tests/pi_runtime/test_memory.py` (11 testes) | partial |
| Skills | Real: seleção por relevância determinística, tracking de uso/success rate | `src/pi_runtime/skills.py`; `tests/pi_runtime/test_skills.py` (12 testes) | partial |
| Delegation (subagents paralelos) | Real: spawn real de processo, live orchestration (list/stop/steer), isolamento de falha | `src/pi_coding_agent/subagent/`; `src/pi_runtime/delegation.py`; `tests/pi_runtime/test_delegation.py` (9 testes, incluindo 1 end-to-end real) | partial |
| Scheduler / jobs recorrentes | Real: one-shot, intervalo fixo (não cron real), retry, persistido, cancelável | `src/pi_runtime/scheduler.py`; `tests/pi_runtime/test_scheduler.py` (17 testes) | partial |
| Execution environments (local/docker/ssh/sandbox) | Real para local/docker/ssh; sandbox explicitamente não implementado (não fake) | `src/pi_runtime/environments.py`; `tests/pi_runtime/test_environments.py` (14 testes) | partial |
| MCP | Real como adapter sobre o Tool Registry; SDK real não instalado (sem servidor MCP configurado) | `src/pi_runtime/mcp.py`; `tests/pi_runtime/test_mcp.py` (13 testes) | not implemented (conexão real) / partial (adapter) |
| Model routing | Real: tier-based, fallback determinístico, budget-aware | `src/pi_runtime/router.py`; `tests/pi_runtime/test_router.py` (16 testes) | partial |
| Verification | Real: Verifier por fase (agent/tool/research/skill), nunca aceita "parece que funcionou" sem checar stop_reason/evidência | `src/pi_runtime/loop.py`, `research.py`, `tools.py`, `learning.py` | partial |
| Cost tracking | Real: $ por 1M tokens, orçamento com enforcement real (Budget.exceeded) | `src/pi_runtime/state.py` (`Budget`), `router.py` (`estimate_cost`) | partial |
| Latency / telemetry | Real: spans com trace_id, duração, custo, reconstrução completa de uma execução | `src/pi_runtime/telemetry.py`; `tests/pi_runtime/test_telemetry.py` (15 testes) | partial |
| Recovery / replanning | Real: Replanner com limite de tentativas, repair steps rastreáveis | `src/pi_runtime/loop.py` (`Replanner`); `tests/pi_runtime/test_loop.py` | partial |
| Observability (JSON export, CLI) | Real: `pi_runtime.cli` com saída JSON em todo comando | `src/pi_runtime/cli.py`; `tests/pi_runtime/test_runtime_cli.py` (11 testes) | partial |

## Por que tudo está marcado `partial` e não `equivalent`/`better`

Cada linha acima tem evidência real do **nosso** lado (código + testes passando). Nenhuma tem
evidência de uma execução comparativa real contra o Hermes nesta sessão — não há acesso a uma
instância do Hermes para rodar o mesmo cenário nos dois sistemas e comparar resultado. Marcar
qualquer uma dessas linhas como `equivalent` ou `better` seria exatamente a "certeza inventada"
que o `plan.md` proíbe explicitamente (seção 6: "reconhecer 'não há evidência suficiente' em
vez de inventar certeza"; seção 23: "nunca declarar 'melhor' sem evidência").

## O que seria necessário para uma badge real

1. Acesso executável ao Hermes Agent (repositório + ambiente configurado).
2. Um conjunto de tarefas idênticas rodadas nos dois sistemas (mesmo objetivo, mesmo
   orçamento).
3. As suites de eval da Fase 16 (`pi_runtime.evals`) aplicadas ao *resultado* de cada sistema,
   produzindo métricas comparáveis (não apenas "funcionou"/"não funcionou").
4. Registro de custo/latência real via `pi_runtime.telemetry` (Fase 15) para ambos.

Nenhum desses quatro pré-requisitos foi satisfeito nesta sessão — este documento registra
honestamente essa lacuna em vez de preenchê-la com uma comparação fabricada.

## Resultado real desta iteração (0-18)

O runtime (`src/pi_runtime/`) evoluiu de zero para 19 módulos cobrindo o ciclo completo
`goal → plan → act → verify → replan | finish`, com Context Engine, Tool Registry + Policy,
Research/Browser/Delegation/Memory/Skills/Model Router/Execution Backends/Sessions+Replay/
MCP/Scheduler/Telemetry/Evals/CLI — **1479 testes passando** na suíte completa do repositório
(incluindo os ~284 específicos de `pi_runtime`), preservando 100% do comportamento existente
de `AgentSession`/`pi_memory`/`pi_coding_agent.subagent` em cada fase (nunca reescrito, sempre
envolvido). Isso é o resultado verificável desta sessão — a comparação contra o Hermes
propriamente dita fica como trabalho futuro explícito, não fabricado.
