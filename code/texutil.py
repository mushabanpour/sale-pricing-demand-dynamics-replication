"""Shared LaTeX helpers for the section generators."""
import re


def firm_names(tex):
    """Display names for the firm codes the results files carry."""
    return tex.replace("MolsonCoors", "Molson Coors")


def place_floats(body, floats):
    """Insert each float immediately after the paragraph that first references it.

    LaTeX can only place a float at or after the point it is declared, so a table
    declared at the end of a section can never appear beside the text that discusses
    it.  Each float is therefore moved next to its first \\ref; any float that is
    never referenced is appended, which is also the signal that something is wrong.
    """
    paras = body.split("\n\n")
    after = {}
    unplaced = []
    for tex in floats:
        m = re.search(r"\\label\{((?:tab|fig):[^}]+)\}", tex)
        lab = m.group(1) if m else None
        hit = None
        if lab:
            for i, p in enumerate(paras):
                if re.search(r"\\ref\{" + re.escape(lab) + r"\}", p):
                    hit = i
                    break
        if hit is None:
            unplaced.append(lab or "<unlabeled>")
        else:
            after.setdefault(hit, []).append(tex)
    out = []
    for i, p in enumerate(paras):
        out.append(p)
        for tex in after.get(i, []):
            out.append(tex)
    out.extend(floats[i] for i, t in enumerate(floats)
               if re.search(r"\\label\{((?:tab|fig):[^}]+)\}", t)
               and re.search(r"\\label\{((?:tab|fig):[^}]+)\}", t).group(1) in unplaced)
    if unplaced:
        print(f"  WARNING: never referenced, appended at end: {unplaced}")
    return "\n\n".join(out)


def place_inputs(body, pairs):
    """Like place_floats, but inserts \\input{Name} for a float that lives in its own
    file.  `pairs` is [(label, name), ...]; the input goes after the paragraph that
    first references \\ref{label}, so float placement is unchanged by the split."""
    paras = body.split("\n\n")
    after, unplaced = {}, []
    for lab, name in pairs:
        hit = next((i for i, p in enumerate(paras)
                    if re.search(r"\\ref\{" + re.escape(lab) + r"\}", p)), None)
        if hit is None:
            unplaced.append(name)
        else:
            after.setdefault(hit, []).append(rf"\input{{{name}}}")
    out = []
    for i, p in enumerate(paras):
        out.append(p)
        out.extend(after.get(i, []))
    out.extend(rf"\input{{{n}}}" for n in unplaced)
    if unplaced:
        print(f"  WARNING: never referenced, appended at end: {unplaced}")
    return "\n\n".join(out)
