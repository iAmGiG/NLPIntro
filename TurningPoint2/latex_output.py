def generate_latex_output(iteration_data, source_tokens, target_tokens):
    latex_content = "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n"
    latex_content += "\\begin{document}\n"

    # Initial Translation Probabilities
    latex_content += "\\section*{Initial Translation Probabilities}\n"
    latex_content += "\\begin{tabular}{lll}\n"
    latex_content += "\\hline\n"
    latex_content += "English Word & Spanish Word & $t(f|e)$ \\\\\n"
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
        latex_content += "\\begin{tabular}{llll}\n"
        latex_content += "\\hline\n"
        latex_content += "English Word & Spanish Word & Normalization Factor & Count($e,f$) \\\\\n"
        latex_content += "\\hline\n"
        count = data['count']
        normalization_factors = data['norm']
        for f in target_tokens:
            norm = normalization_factors[f]
            for e in source_tokens:
                cnt = count[e][f]
                latex_content += f"{e} & {f} & {norm:.4f} & {cnt:.4f} \\\\\n"
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"

        # Maximization Step
        latex_content += "\\subsection*{Maximization Step}\n"
        latex_content += "\\begin{tabular}{lll}\n"
        latex_content += "\\hline\n"
        latex_content += "English Word & Spanish Word & Updated $t(f|e)$ \\\\\n"
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
