# Script pour corriger la fonction run_backtest dans model.py

with open("model.py", "r") as f:
    lines = f.readlines()

# Trouver et remplacer la ligne problématique
new_lines = []
for i, line in enumerate(lines):
    if 'backtest_df["Trend_Filter"] = (' in line:
        # Remplacer les 3 prochaines lignes
        new_lines.append('        backtest_df["Trend_Filter"] = (\n')
        new_lines.append(
            '            backtest_df["SPY_Close"].squeeze() > backtest_df["SMA50"].squeeze()\n'
        )
        new_lines.append("        ).astype(int)\n")
        # Sauter les 2 lignes suivantes de l'original
        skip = 0
        for j in range(i + 1, min(i + 3, len(lines))):
            if ").astype(int)" in lines[j]:
                skip = j - i
                break
        if skip > 0:
            continue
    else:
        new_lines.append(line)

with open("model.py", "w") as f:
    f.writelines(new_lines)

print("✓ model.py corrigé")
