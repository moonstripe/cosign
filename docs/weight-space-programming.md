# Weight-space programming for the mood wheel

Each mood axis on the radial wheel (tense ↔ calm, …) is backed by a **dial**: a
low-rank edit to LFM3-1.7B's weights that can be turned between 0 (the
original model) and 1, token by token. Dials are not trained. An LLM
"programmer" writes each one from a plain-English spec. It reads the weights,
runs a limited number of forward passes, and edits the weights in closed form.
This is the setup of *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang,
Khattab; MIT CSAIL 2026), Experiment 8.

```mermaid
flowchart TB
    subgraph BUILD["Offline: building a dial (one run per mood axis)"]
        direction TB
        SPEC["Task spec<br/>steer/wsp/tasks/&lt;axis&gt;.md<br/>'make the model write tensely, not calmly'"]
        PROG["Programmer<br/>(LLM agent: reasons, writes Python)"]
        SPEC --> PROG

        subgraph LAB["wsp lab (steer/wsp/lab.py, :8790) — LFM3-1.7B loaded once"]
            direction TB
            KERNEL["Kernel over the weights<br/>load(name) · names() · acts(k) · gens(k)<br/>save_dial() · submit_dial()<br/><i>no model object, no gradients</i>"]
            FWD["Forward-pass tool<br/>budget: 15 passes"]
            GEN["generate<br/>≤16 prompts × ≤16 samples<br/>original vs candidate dials, paired seeds"]
            READ["read (teacher-forced)<br/>≤256 prefix+continuation pairs<br/>log-probs · residual means [24×2048]<br/>w2 inputs [7680]"]
            FWD --> GEN
            FWD --> READ
            LOG[("wsp_runs/&lt;axis&gt;/<br/>tool_calls.jsonl · requests/ · work/<br/>notes.md · report.md")]
            KERNEL -.logged.-> LOG
            FWD -.logged.-> LOG
        end

        PROG -- "uv run wsp kernel" --> KERNEL
        PROG -- "uv run wsp forward (costs 1 pass)" --> FWD
        GEN -- "texts the programmer reads & judges" --> PROG
        READ -- "activations → acts(k)" --> KERNEL

        subgraph LOOP["Typical programmer loop (Exp. 8 pattern, not prescribed)"]
            direction LR
            L1["1 · elicit<br/>tense vs plain continuations<br/>of the same openings"] --> L2["2 · read both without the instruction<br/>→ matched activation contrast<br/>→ direction u per layer"]
            L2 --> L3["3 · localise writers<br/>MLP w2 columns that write u<br/>(top-k neurons)"]
            L3 --> L4["4 · write the edit<br/>ΔW = u ⊗ (mask ⊙ uᵀW)<br/>(directional gain)"]
            L4 --> L5["5 · test<br/>dose–response · random control<br/>held-out paired generations"]
            L5 -. revise .-> L2
        end
        PROG --- LOOP

        DIAL[("moods/tense__calm.pt<br/>deltas: (module, A, B)<br/>W' = W + α·A·Bᵀ")]
        KERNEL -- "submit_dial()" --> DIAL
    end

    subgraph RUN["Online: co-writing (steer/server.py + frontend)"]
        direction TB
        WHEEL["Radial wheel<br/>direction → axis · distance → dial 0–1"]
        ENGINE["Engine.set_signposts()<br/>(concept, anti) has a dial? → α<br/>no dial → no steering"]
        RACK["DialRack (steer/dials.py)<br/>forward hooks on the edited nn.Linear:<br/>y += α · (x·B)·Aᵀ"]
        MODEL["LFM3-1.7B<br/>next token"]
        WHEEL -- "websocket: dial settings per request" --> ENGINE
        ENGINE --> RACK --> MODEL
        MODEL -- "α can change on the very next token" --> WHEEL
    end

    DIAL -- "hot-reloaded (mtime)" --> ENGINE
```

## Pieces

| Piece | File | Role |
| --- | --- | --- |
| Task spec | `steer/wsp/tasks/<axis>.md` | Natural-language goal, tools, rules, deliverables |
| Lab | `steer/wsp/lab.py` (`uv run wsp-lab --run <axis> --budget 15`) | Holds the model; kernel + budgeted forward passes; logs everything |
| Programmer CLI | `steer/wsp/cli.py` (`uv run wsp status\|kernel\|forward`) | The only way the programmer touches the lab |
| Dial format + runtime | `steer/dials.py` | `Dial`, `DialRack` (hooks, per-token α), `bake()` for export |
| Engine integration | `steer/engine.py` | Wheel settings whose (concept, anti) has a dial turn it; there is no other steering |
| Run logs | `wsp_runs/<axis>/` | Every kernel call and forward pass, plus the programmer's notes and report |

## Rules the programmer works under

- No gradients or training: no backward passes, autograd, optimisers, or iterative loss fitting.
  Closed-form linear algebra on weights and recorded activations is allowed.
- The model runs only through the forward-pass tool, which has a fixed budget.
- The embedding, the output head and the final norm are off limits.
- α = 0 is exactly the original model (verified), and α = 1 is the strongest setting the programmer judged
  still fluent.

## Results

The four shipped dials, their evidence and their limits are in [METHODOLOGY.md](../METHODOLOGY.md) §4. Each run's
own report is in `wsp_runs/<axis>/report.md`.
