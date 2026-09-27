"""
Synthetic data generator that mimics the challenge format and noise patterns.

PURPOSE: integration / scale testing ONLY. Numbers measured on synthetic data are NOT
estimates of real-data performance. It reproduces the *structure* reported by the real
data audit (multi-match distribution, ~5.6% singletons, ~26% unmatched S2/S3 distractors,
~30% duplicate S1 names from chains, ~3.4% empty S2/S3 addresses, heavier punctuation
noise in S2, France only in test) so that code paths, file formats, memory behaviour and
metric plumbing can be exercised end-to-end without the real files.
"""
from __future__ import annotations

import argparse
import os
import random
import string

US_WORDS = ("acme summit liberty eagle pioneer golden harbor maple cedar granite atlas beacon "
            "keystone northstar redwood silver blue ridge valley river lakeside prairie frontier "
            "heritage patriot evergreen crescent sterling apex vertex horizon jackson smith johnson "
            "miller davis wilson moore taylor anderson thomas martin garcia clark lewis walker").split()
US_TYPES = ("dental plumbing roofing bakery pharmacy auto repair consulting logistics insurance "
            "realty construction electric landscaping cafe diner grill fitness salon cleaning "
            "hardware printing marketing solutions services partners group holdings").split()
US_SUFFIX = ["Inc", "Inc.", "LLC", "L.L.C.", "Corp", "Corporation", "Co", "Company", "Ltd", ""]
US_STREETS = ("Main Oak Pine Maple Cedar Elm Washington Lake Hill Park Sunset Lincoln Madison "
              "Jefferson Franklin Highland Church Market Spring Ridge").split()
US_STYPE = [("Street", "St"), ("Avenue", "Ave"), ("Road", "Rd"), ("Boulevard", "Blvd"),
            ("Drive", "Dr"), ("Lane", "Ln"), ("Court", "Ct")]
US_CITIES = [("Springfield", "IL"), ("Austin", "TX"), ("Denver", "CO"), ("Columbus", "OH"),
             ("Portland", "OR"), ("Madison", "WI"), ("Raleigh", "NC"), ("Tampa", "FL"),
             ("Phoenix", "AZ"), ("Boise", "ID"), ("Omaha", "NE"), ("Richmond", "VA")]

IN_WORDS = ("sharma gupta agarwal patel singh kumar verma mehta joshi reddy iyer nair shree sri "
            "ganesh lakshmi balaji krishna durga sai om jai bharat hind national royal new "
            "super star modern universal").split()
IN_TYPES = ("traders enterprises textiles jewellers medicals electronics hardware sweets "
            "motors agencies industries exports foods chemicals steels constructions "
            "infotech solutions associates").split()
IN_SUFFIX = ["Pvt Ltd", "Private Limited", "Pvt. Ltd.", "LLP", "Ltd", "Limited", ""]
IN_AREAS = ("Gandhi Nagar", "Nehru Road", "MG Road", "Station Road", "Civil Lines", "Sector 15",
            "Rajendra Nagar", "Malviya Nagar", "Shastri Colony", "Industrial Area", "Tilak Marg")
IN_LANDMARKS = ("SBI ATM", "Bus Stand", "Railway Station", "Hanuman Mandir", "Post Office",
                "Govt School", "City Hospital")
IN_CITIES = [("Jaipur", "Rajasthan", "302"), ("Pune", "Maharashtra", "411"),
             ("Lucknow", "Uttar Pradesh", "226"), ("Indore", "Madhya Pradesh", "452"),
             ("Surat", "Gujarat", "395"), ("Kochi", "Kerala", "682"),
             ("Nagpur", "Maharashtra", "440"), ("Patna", "Bihar", "800")]

FR_WORDS = ("martin bernard dubois durand lefebvre moreau laurent simon michel garnier "
            "fontaine rousseau vincent muller lefevre boulanger belle petit grand soleil "
            "provence atlantique alpes").split()
FR_TYPES = ("boulangerie patisserie pharmacie garage coiffure boucherie librairie fleuriste "
            "transports conseil immobilier batiment restaurant cafe informatique").split()
FR_SUFFIX = ["SARL", "SAS", "S.A.S.", "SA", "EURL", "SASU", ""]
FR_STYPE = [("Rue", "R."), ("Avenue", "Av."), ("Boulevard", "Bd"), ("Place", "Pl."),
            ("Chemin", "Ch."), ("Allée", "All.")]
FR_STREETS = ("de la République", "Victor Hugo", "Jean Jaurès", "Pasteur", "de la Gare",
              "du Général Leclerc", "des Écoles", "Voltaire", "de Paris", "Émile Zola")
FR_CITIES = [("Lyon", "69"), ("Marseille", "13"), ("Toulouse", "31"), ("Nantes", "44"),
             ("Bordeaux", "33"), ("Lille", "59"), ("Rennes", "35"), ("Nice", "06")]

MATCH_COUNT_DIST = [(0, .0558), (1, .054), (2, .17), (3, .2405), (4, .2194), (5, .1459),
                    (6, .0747), (7, .029), (8, .0085), (9, .0019), (10, .0003)]


def _pick_count(r):
    x, acc = r.random(), 0.0
    for c, p in MATCH_COUNT_DIST:
        acc += p
        if x < acc:
            return c
    return 3


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)

    # ---------------- base entity ----------------
    def name(self, country, chain_pool):
        r = self.r
        if chain_pool and r.random() < 0.30:
            return r.choice(chain_pool)
        if country == "US":
            w, t, s = US_WORDS, US_TYPES, US_SUFFIX
        elif country == "India":
            w, t, s = IN_WORDS, IN_TYPES, IN_SUFFIX
        else:
            w, t, s = FR_WORDS, FR_TYPES, FR_SUFFIX
        k = r.choice([1, 1, 2])
        core = " ".join(x.title() for x in r.sample(w, k))
        if r.random() < 0.8:
            core += " " + r.choice(t).title()
        if r.random() < 0.15:
            core += " & " + r.choice(w).title()
        if r.random() < 0.15:  # rare coined token -> unique names
            core += " " + "".join(r.choice(string.ascii_lowercase) for _ in range(6)).title()
        suf = r.choice(s)
        return (core + " " + suf).strip()

    def address(self, country):
        r = self.r
        if country == "US":
            city, st = r.choice(US_CITIES)
            full, _ = r.choice(US_STYPE)
            return {"num": str(r.randint(1, 9999)), "street": r.choice(US_STREETS), "stype": full,
                    "city": city, "region": st, "postal": f"{r.randint(10000, 99999)}"}
        if country == "India":
            city, state, p3 = r.choice(IN_CITIES)
            return {"num": f"{r.randint(1, 400)}", "street": r.choice(IN_AREAS), "stype": "",
                    "city": city, "region": state, "postal": f"{p3}{r.randint(0, 999):03d}",
                    "landmark": r.choice(IN_LANDMARKS)}
        city, dep = r.choice(FR_CITIES)
        full, _ = r.choice(FR_STYPE)
        return {"num": str(r.randint(1, 250)), "street": r.choice(FR_STREETS), "stype": full,
                "city": city, "region": "", "postal": f"{dep}{r.randint(0, 999):03d}"}

    def render_addr(self, a, country, noisy=False):
        r = self.r
        stype = a["stype"]
        if noisy and stype and r.random() < 0.5:
            table = US_STYPE if country == "US" else FR_STYPE
            for full, ab in table:
                if full == stype:
                    stype = ab
        if country == "US":
            parts = [f"{a['num']} {a['street']} {stype}", a["city"], f"{a['region']} {a['postal']}"]
        elif country == "India":
            parts = [f"{'Plot No. ' if r.random() < .3 else ''}{a['num']}", a["street"],
                     a["city"], f"{a['region']} - {a['postal']}"]
            if noisy and r.random() < 0.3:
                parts.insert(2, "Near " + a["landmark"])
        else:
            parts = [f"{a['num']} {stype} {a['street']}", f"{a['postal']} {a['city']}"]
        if noisy:
            if r.random() < 0.25:
                parts = parts[:-1]  # drop postal/state component
            if r.random() < 0.10 and len(parts) > 2:
                r.shuffle(parts)
        return ", ".join(p for p in parts if p)

    # ---------------- noise ----------------
    def typo(self, s):
        r = self.r
        if len(s) < 5:
            return s
        i = r.randrange(1, len(s) - 1)
        op = r.random()
        if op < 0.33:
            return s[:i] + s[i + 1:]
        if op < 0.66:
            return s[:i] + s[i + 1] + s[i] + s[i + 2:]
        return s[:i] + r.choice(string.ascii_lowercase) + s[i + 1:]

    def noisy_name(self, n, source):
        r = self.r
        s = n
        rep = [("Private Limited", "Pvt Ltd"), ("Pvt Ltd", "Private Limited"), ("Corporation", "Corp"),
               ("Corp", "Corporation"), ("Inc", "Incorporated"), ("Company", "Co"), ("Limited", "Ltd"),
               (" & ", " and "), (" and ", " & ")]
        for a, b in rep:
            if a in s and r.random() < 0.35:
                s = s.replace(a, b, 1)
                break
        toks = s.split()
        if len(toks) > 2 and r.random() < 0.25:  # drop legal suffix
            toks = toks[:-1]
        if len(toks) > 2 and r.random() < 0.08:  # transposition
            i = r.randrange(len(toks) - 1)
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
        s = " ".join(toks)
        if r.random() < 0.15:
            s = self.typo(s)
        if r.random() < 0.2:
            s = s.upper() if r.random() < .5 else s.lower()
        punct_p = 0.45 if source == 2 else 0.2
        if r.random() < punct_p:
            s = r.choice([s + ".", s.replace(" ", ", ", 1), "\"" + s + "\"", s.replace(" ", " - ", 1),
                          s + ","])
        return s


def generate(out_dir, n_s1_train=20000, n_s1_test=15000, seed=7):
    for split, n_s1 in (("train", n_s1_train), ("test", n_s1_test)):
        g = Gen(seed + (0 if split == "train" else 1))
        r = g.r
        countries = (["US"] * 60 + ["India"] * 40) if split == "train" else \
                    (["US"] * 38 + ["India"] * 47 + ["France"] * 15)
        chains = {c: [g.name(c, None) for _ in range(max(5, n_s1 // 400))] for c in ("US", "India", "France")}
        s1, s2, s3, gt = [], [], [], []
        c2 = c3 = 0

        def add(src, rec):
            nonlocal c2, c3
            if src == 2:
                c2 += 1
                eid = f"S2-{split[:2]}{c2:08d}"
                s2.append((eid,) + rec)
            else:
                c3 += 1
                eid = f"S3-{split[:2]}{c3:08d}"
                s3.append((eid,) + rec)
            return eid

        for i in range(n_s1):
            c = r.choice(countries)
            name = g.name(c, chains[c])
            a = g.address(c)
            s1_id = f"S1-{split[:2]}{i + 1:08d}"
            s1.append((s1_id, name, g.render_addr(a, c), c))
            k = _pick_count(r)
            matched = []
            for _ in range(k):
                src = 2 if r.random() < 0.484 else 3
                addr = "" if r.random() < 0.034 else g.render_addr(a, c, noisy=True)
                matched.append(add(src, (g.noisy_name(name, src), addr, c)))
            if k == 0 and r.random() < 0.5:
                # singleton with a tempting same-name, different-location distractor
                a2 = g.address(c)
                add(r.choice((2, 3)), (g.noisy_name(name, 2), g.render_addr(a2, c, True), c))
            gt.append((s1_id, ",".join(matched)))
        # unmatched distractors (~26% of S2/S3)
        n_dis = int(0.35 * (len(s2) + len(s3)))
        for _ in range(n_dis):
            c = r.choice(countries)
            src = r.choice((2, 3))
            if r.random() < 0.4:  # same chain, other location: hard negative
                nm = g.noisy_name(r.choice(chains[c]), src)
            else:
                nm = g.noisy_name(g.name(c, None), src)
            add(src, (nm, "" if r.random() < 0.034 else g.render_addr(g.address(c), c, True), c))
        r.shuffle(s2)
        r.shuffle(s3)
        d = os.path.join(out_dir, split)
        os.makedirs(d, exist_ok=True)
        hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        for nm, rows in (("source1", s1), ("source2", s2), ("source3", s3)):
            with open(os.path.join(d, f"{split}_{nm}.tsv"), "w", encoding="utf-8") as f:
                f.write(hdr)
                for row in rows:
                    f.write("\t".join(x.replace("\t", " ") for x in row) + "\n")
        if split == "train":
            with open(os.path.join(d, "train_ground_truth.tsv"), "w", encoding="utf-8") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                for a, b in gt:
                    f.write(f"{a}\t{b}\n")
        else:  # hidden answer key for synthetic-only end-to-end scoring
            with open(os.path.join(out_dir, "_synthetic_test_answer_key.tsv"), "w", encoding="utf-8") as f:
                f.write("source1_entity_id\tmatched_entity_ids\n")
                for a, b in gt:
                    f.write(f"{a}\t{b}\n")
        print(f"[synth] {split}: S1={len(s1):,} S2={len(s2):,} S3={len(s3):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=15000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    generate(a.out, a.n_train, a.n_test, a.seed)
