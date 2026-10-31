p = "src/Modules/Diffusion/sdvkgen.cpp"
lines = open(p, encoding="utf-8").read().split("\n")
out = []
i = 0
while i < len(lines):
    l = lines[i]
    out.append(l)
    i += 1
s = "\n".join(out)
s = s.replace('(out @%.1f)\n", p.c_str()', '(out @%.1f) | ", p.c_str()')
open(p, "w", encoding="utf-8").write(s)
print("ok")
