"""Self-test of pcr_final (no torch, about a minute): python test_pcr_final.py -> every line PASS.

Fake data with judgment-style headers (DATE OF JUDGMENT, CITATION, CITATOR INFO). Each query cites older cases of
its topic and also copies phrases from newer cases of its topic (which it cannot cite), so the date filter has
something to remove; queries of one topic cite overlapping cases, so the similar-queries votes carry signal; fake
runs of two models give noisy but informative scores.
"""
import json
import os
import random
import sys
import tempfile
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pcr_data as PD  # noqa: E402
import pcr_final as PF  # noqa: E402

AGGS = ("mean", "maxsim", "maxsim_sym", "max", "first")


def _header(rng, y, own_page=None, citator=(), cit=True, date=True):
    h = "PETITIONER: <ORG> Vs. RESPONDENT: <NAME> "
    if date:
        h += f"DATE OF JUDGMENT: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{y} "
    h += "BENCH: <NAME> "
    if cit:
        h += f"CITATION: {y} AIR {own_page or rng.randint(2000, 2999)} {y} SCR (2) {rng.randint(1, 900)} "
    entries = list(citator) + [(min(2025, y + 9), rng.randint(1000, 1999)), (min(2025, y + 15), rng.randint(1000, 1999))]
    rng.shuffle(entries)
    h += "CITATOR INFO : " + " ".join(f"{rng.choice('RFED')} {a} SC{b} ({rng.randint(1, 20)})" for a, b in entries) + " "
    return h + "ACT: Indian Act, 1950 HEADNOTE: "


def make_fake_final(root, seed=0, n_cases=320, topics=6):
    """Cases, then queries; 60% of the (query, cited case) pairs put the query's citation into the cited case's
    CITATOR INFO (as in real headnotes); everything else in the citator lists is noise."""
    rng = random.Random(seed)
    vocab = {t: [f"t{t}w{i}" for i in range(60)] for t in range(topics)}
    common = [f"g{i}" for i in range(200)]
    cases = []
    for j in range(n_cases):
        t, y = rng.randrange(topics), rng.randint(1950, 2010)
        words = [rng.choice(vocab[t]) if rng.random() < 0.4 else rng.choice(common) for _ in range(rng.randint(150, 260))]
        cases.append(dict(id=f"c{j}", t=t, y=y, words=words, citator=[], cit=rng.random() < 0.9,
                          date=rng.random() < 0.9))
    pools = {"train": cases[:230], "dev": cases[80:320], "test": cases[40:300]}
    gold_all, base, queries, page = {}, os.path.join(root, "Phase_7_PCR"), {}, 0
    for split, nq in (("train", 70), ("dev", 30), ("test", 30)):
        pool = pools[split]
        qs = []
        for i in range(nq):
            t, Y = rng.randrange(topics), rng.randint(1975, 2000)
            older = [c for c in pool if c["t"] == t and c["y"] < Y]
            newer = [c for c in pool if c["t"] == t and c["y"] > Y]
            if len(older) < 3:
                continue
            older.sort(key=lambda c: c["id"])
            rel = rng.sample(older[:max(3, len(older) * 2 // 3)], min(len(older), rng.randint(3, 7)))
            words = [rng.choice(vocab[t]) if rng.random() < 0.4 else rng.choice(common) for _ in range(200)]
            for c in rel + rng.sample(newer, min(2, len(newer))):            # cited cases + newer decoys
                for _ in range(rng.choice([0, 1, 2, 3])):
                    s = rng.randint(0, len(c["words"]) - 6)
                    p = rng.randint(0, len(words))
                    words[p:p] = c["words"][s:s + 5] + (["<CITATION>"] if c in rel else [])
            page += 1
            for c in rel:
                if rng.random() < 0.6:
                    c["citator"].append((Y, page))
            qid = f"{split[:2]}q{i}"
            q = dict(id=qid, text=_header(rng, Y, own_page=page) + " ".join(words))
            if split != "test":
                q["relevant_candidates"] = [c["id"] for c in rel]
            gold_all[qid] = [c["id"] for c in rel]
            qs.append(q)
        queries[split] = qs
    for c in cases:
        c["text"] = _header(rng, c["y"], citator=c["citator"], cit=c["cit"], date=c["date"]) + " ".join(c["words"])
    for split in ("train", "dev", "test"):
        d = os.path.join(base, split.capitalize())
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{split}.json"), "w") as f:
            json.dump(queries[split], f)
        with open(os.path.join(d, f"{split}_candidates.json"), "w") as f:
            json.dump([dict(id=c["id"], text=c["text"]) for c in pools[split]], f)
    return base, gold_all


def make_fake_runs(root, L, gold_all, seed=0, zipped=False):
    rng = np.random.default_rng(seed)
    for model in ("llama", "moe"):
        out_root = os.path.join(root, f"pcr_results_{model}", "pcr_runs")
        for tech, signal in (("full", 2.0), ("lorma", 1.6), ("zeroshot", 0.0)):
            d = os.path.join(out_root, f"{model}_{tech}")
            os.makedirs(d, exist_ok=True)
            for s in ("dev", "test"):
                sp = L[s]
                G = np.zeros((len(sp["qids"]), len(sp["pids"])), dtype=np.float32)
                pos = {c: j for j, c in enumerate(sp["pids"])}
                for i, q in enumerate(sp["qids"]):
                    for c in gold_all[q]:
                        G[i, pos[c]] = 1
                base = rng.normal(0, 1, G.shape).astype(np.float32) + signal * G
                np.savez_compressed(os.path.join(d, f"scores_{s}.npz"),
                                    **{a: base + rng.normal(0, 0.3, G.shape).astype(np.float32) for a in AGGS})
            with open(os.path.join(d, "ids.json"), "w") as f:
                json.dump({s: dict(qids=L[s]["qids"], pids=L[s]["pids"]) for s in ("dev", "test")}, f)
            with open(os.path.join(d, "metrics.json"), "w") as f:
                json.dump(dict(model=model, technique=tech), f)
        if zipped:
            z = os.path.join(root, f"pcr_results_{model}.zip")
            with zipfile.ZipFile(z, "w") as zf:
                for r, _, files in os.walk(os.path.join(root, f"pcr_results_{model}")):
                    for fn in files:
                        full = os.path.join(r, fn)
                        zf.write(full, os.path.relpath(full, os.path.join(root, f"pcr_results_{model}")))
            import shutil
            shutil.rmtree(os.path.join(root, f"pcr_results_{model}"))


def test_doc_date():
    d = PF.doc_date("PETITIONER: X DATE OF JUDGMENT: 08/12/1965 BENCH: Y CITATION: 1966 AIR 1053 CITATOR INFO : "
                    "R 1986 SC1483 (4) ACT: Income-tax Act, 1922")
    assert d["ord"] == 19651208 and d["lo"] == d["hi"] == 1965, d
    assert PF.doc_date("Date of Judgment: 12th March, 1999. The appeal")["ord"] == 19990312
    assert PF.doc_date("DATE OF JUDGMENT: March 5, 2001 BENCH")["ord"] == 20010305
    c = PF.doc_date("PETITIONER: X CITATION: 1972 AIR 10 CITATOR INFO : R 1990 SC 5 (3) ACT: x")
    assert c["src"] == "citation year" and (c["lo"], c["hi"]) == (1971, 1972), c
    own, tor, men = PF.cite_keys("DATE OF JUDGMENT: 08/12/1965 CITATION: 1966 AIR 1053 1966 SCR (2) 944 CITATOR INFO : "
                                 "F 1967 SC 819 (5) E 1969 SC1160 (5) ACT: x as held in AIR 1955 SC 123 and 1960 AIR 7")
    assert own == {(1966, 1053)} and tor == {(1967, 819), (1969, 1160)} and men == {(1955, 123), (1960, 7)}, (own, tor, men)
    assert PF.doc_date("DATE OF JUDGMENT07/10/1983 BENCH: x CITATION: 1984 SCR (1) 347")["ord"] == 19831007
    assert PF.doc_date("as held in <CITATION> 1999 the court")["src"] != "citation year"
    cx = PF.contexts("a b c d e f <CITATION> g h i j k l m n <CITATION> o", width=3)
    assert cx == ["d e f g h i"], cx                    # the second context has only 4 words, so it is skipped
    t = PF.doc_date("in 1950 the Act ... CITATOR INFO : R 1999 SC 5 (3) ACT: decided after 1961")
    assert t["src"].startswith("latest year") and t["lo"] == 1961 and t["hi"] is None, t     # citator ignored
    q = [PF.doc_date("DATE OF JUDGMENT: 01/01/1980 x")]
    p = [PF.doc_date("DATE OF JUDGMENT: 02/01/1980 x"), PF.doc_date("DATE OF JUDGMENT: 31/12/1979 x"),
         PF.doc_date("CITATION: 1982 AIR 1 x"), PF.doc_date("CITATION: 1981 AIR 1 x"), PF.doc_date("nothing")]
    later, gap, known = PF.date_matrices(q, p)
    assert later[0].tolist() == [True, False, True, False, False], later      # 1981 citation may be a 1980 case
    assert known[0, 4] == 0 and known[0, 1] == 1


def test_main_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root, gold_all = make_fake_final(tmp)
        data = PD.load_data(root, fetch=False)
        L = PD.lexical(data, ns=(1, 2, 3, 4, 5), log=lambda *a: None)
        lex = os.path.join(tmp, "lex.pkl")
        PD.save_lexical(L, lex)
        res = os.path.join(tmp, "results")
        make_fake_runs(res, L, gold_all, zipped=True)
        out = os.path.join(tmp, "final")
        logs = []
        rows = PF.main(results_roots=[res], out_dir=out, data_root=root, lexical_path=lex, fetch=False,
                       log=logs.append)
        text = "\n".join(map(str, logs))
        assert "[runs] 6 runs" in text and "cited cases wrongly marked later: 0 of" in text, text[:3000]
        assert "query in the candidate's CITATOR INFO: finds" in text and "precision 100.0%" in text, text[:4000]
        assert len(rows) == 2 * 4 * 2 * 3 and rows[0]["f1"] >= rows[-1]["f1"]
        assert max(r["f1"] for r in rows if r["use_cit"]) >= max(r["f1"] for r in rows if not r["use_cit"]) - 0.01
        assert "[contexts] dev:" in text and "alone: ctx_max_2" in text, text[:4000]
        best_lex = max(r["f1"] for r in rows if r["set"] == "lexical" and not r["use_cit"])
        best_dense = max(r["f1"] for r in rows if r["set"] != "lexical" and not r["use_cit"])
        assert best_dense > best_lex, (best_dense, best_lex)
        lf = {r["name"]: r for r in rows}
        assert lf["lexical_nocitator_datefilter_logreg"]["f1"] >= lf["lexical_nocitator_datefeature_logreg"]["f1"] - 0.02
        subs = sorted(f for f in os.listdir(out) if f.endswith(".zip"))
        assert len(subs) in (4, 5) and all(f.startswith("final_") and "__dev0." in f for f in subs), subs
        assert any("nocitator" in f for f in subs), subs
        pred = json.loads(zipfile.ZipFile(os.path.join(out, subs[0])).read("prediction.json"))
        pool = {d["id"] for d in data["test"][1]}
        assert len(pred) == len(data["test"][0]) and all(len(v) == 7 and set(v) <= pool for v in pred.values())
        assert os.path.exists(os.path.join(out, "final_results.txt"))
        print(f"    best: {rows[0]['name']} {rows[0]['f1']:.3f}; best lexical only {best_lex:.3f}")


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
