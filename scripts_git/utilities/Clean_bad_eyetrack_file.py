from pathlib import Path
import io, re, pandas as pd

# path = Path(r"K:\BramBurleson\000_datasets_and_scripts\20250603_NAT_V2_fMRI_pilots\NAT_V2_fMRI_pilots\data\eyetrack\sub-03\single_stream\raw\backup\20250520_sub_03.txt")
path = Path(r"/mnt/ilaria_group/BramBurleson/000_datasets_and_scripts/20250603_NAT_V2_fMRI_pilots/NAT_V2_fMRI_pilots/data/eyetrack/sub-07/single_stream/raw/20250528_sub_07.txt")
clean_lines = []
header = None

with path.open(encoding="utf-8") as f:
    for ln in f:
        tag = ln.split("\t", 1)[0].strip()

        if tag == "5":                           # header row → just remember it
            header = ln.strip().split("\t")
            header = header[:-1]  # drop last column
            # header = [col.strip() for col in re.sub(r"\s{2,}", "\t", ln.strip()).split("\t")]

            # header = [col.strip() for col in re.sub(r"\s{2,}", "\t", ln.strip()).split("\t")][1:]


            # clean_lines.append(ln.strip())   # keep it for output
            # print(ln.strip())            

        elif tag in {"10", "12"}:                # data rows we want
            # some tag-12 rows use spaces – normalise them:

            ln = re.sub(r"\s{2,}", "\t", ln.strip())
            parts = ln.split("\t")
            if len(parts) == 28:
                parts = parts[:-1]  # drop 28th field
            clean_lines.append("\t".join(parts))
            # clean_lines.append(ln)

# build an in-memory buffer that starts with data only (header comes later)
buf = io.StringIO("\n".join(clean_lines))

df = pd.read_csv(buf, sep="\t", header=None, engine="python")
df.columns = header                    # apply header from tag-5 row

filename = r"/mnt/ilaria_group/BramBurleson/000_datasets_and_scripts/20250603_NAT_V2_fMRI_pilots/NAT_V2_fMRI_pilots/data/eyetrack/sub-07/single_stream/raw/20250528_sub_07_clean.txt"
df.to_csv(filename, sep="\t", index=False)

