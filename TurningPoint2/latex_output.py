def generate_latex_output(iteration_data, source_tokens, target_tokens):
    latex_content = "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n"
    latex_content += "\\begin{document}\n"

    # Initialization Section
    latex_content += "\\section*{Initialization}\n"
    latex_content += "The initial translation probabilities are set uniformly:\n"
    latex_content += "\\begin{tabular}{lll}\n"
    latex_content += "\\hline\n"
    latex_content += "English Word ($e_i$) & Spanish Word ($f_j$) & $t(f|e)$ \\\\\n"
    latex_content += "\\hline\n"
    initial_probs = iteration_data[0]['t']
    for e in source_tokens:
        for f in target_tokens:
            prob = initial_probs[f][e]
            latex_content += f"{e} & {f} & {prob:.4f} \\\\\n"
    latex_content += "\\hline\n"
    latex_content += "\\end{tabular}\n"

    # Iterations
    for i, data in enumerate(iteration_data, 1):
        latex_content += f"\\section*{{Iteration {i}}}\n"

        # Expectation Step
        latex_content += "\\subsection*{Expectation Step}\n"
        latex_content += "For each target word $f_j$, compute the normalization factor $s_{total}(f_j)$ and calculate $\\delta(e_i, f_j)$:\n"
        latex_content += "\\begin{tabular}{lllll}\n"
        latex_content += "\\hline\n"
        latex_content += "Spanish Word ($f_j$) & $s_{total}(f_j)$ & English Word ($e_i$) & $t(f_j|e_i)$ & $\\delta(e_i, f_j)$ \\\\\n"
        latex_content += "\\hline\n"
        count = data['count']
        normalization_factors = data['norm']
        delta_values = data['delta']
        for f in target_tokens:
            s_total = normalization_factors[f]
            for e in source_tokens:
                delta = delta_values[e][f]
                prob = data['t'][f][e]
                latex_content += f"{f} & {s_total:.4f} & {
                    e} & {prob:.4f} & {delta:.4f} \\\\\n"
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"

        # Counts Table
        latex_content += "\\subsection*{Counts and Totals}\n"
        latex_content += "\\begin{tabular}{llll}\n"
        latex_content += "\\hline\n"
        latex_content += "English Word ($e_i$) & Spanish Word ($f_j$) & Count($c(e_i, f_j)$) & Total($total(e_i)$) \\\\\n"
        latex_content += "\\hline\n"
        total_e = data['total_e']
        for e in source_tokens:
            for f in target_tokens:
                cnt = count[e][f]
                total = total_e[e]
                latex_content += f"{e} & {f} & {cnt:.4f} & {total:.4f} \\\\\n"
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"

        # Maximization Step
        latex_content += "\\subsection*{Maximization Step}\n"
        latex_content += "Updated translation probabilities:\n"
        latex_content += "\\begin{tabular}{lll}\n"
        latex_content += "\\hline\n"
        latex_content += "English Word ($e_i$) & Spanish Word ($f_j$) & Updated $t(f_j|e_i)$ \\\\\n"
        latex_content += "\\hline\n"
        t = data['t']
        for e in source_tokens:
            for f in target_tokens:
                prob = t[f][e]
                latex_content += f"{e} & {f} & {prob:.4f} \\\\\n"
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"

    latex_content += "\\end{document}"
    return latex_content
