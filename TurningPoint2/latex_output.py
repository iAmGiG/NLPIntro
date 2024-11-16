# latex_output.py
def generate_latex_output(iteration_data, source_tokens, target_tokens):
    """
    Generates LaTeX formatted output for the EM algorithm steps.
    """
    latex_content = "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n"
    latex_content += "\\begin{document}\n"

    # Initial Translation Probabilities
    latex_content += "\\section*{Initial Translation Probabilities}\n"
    latex_content += "Assuming uniform distribution:\n"
    latex_content += "\\begin{tabular}{ll}\n"
    latex_content += "\\hline\n"
    latex_content += "English Word & Spanish Word & $t(f|e)$ \\\\\n"
    latex_content += "\\hline\n"
    initial_probs = iteration_data[0]['t']
    for f in initial_probs:
        for e in initial_probs[f]:
            prob = initial_probs[f][e]
            latex_content += f"{e} & {f} & {prob:.4f} \\\\\n"
    latex_content += "\\hline\n"
    latex_content += "\\end{tabular}\n"

    # Iterations
    for i, data in enumerate(iteration_data, 1):
        latex_content += f"\\section*{{Iteration {i}}}\n"

        # Expectation Step
        latex_content += "\\subsection*{Expectation Step}\n"
        latex_content += "Computed Counts:\n"
        latex_content += "\\begin{tabular}{lll}\n"
        latex_content += "\\hline\n"
        latex_content += "English Word & Spanish Word & Count($e,f$) \\\\\n"
        latex_content += "\\hline\n"
        count = data['count']
        for e in count:
            for f in count[e]:
                cnt = count[e][f]
                latex_content += f"{e} & {f} & {cnt:.4f} \\\\\n"
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"

        # Maximization Step
        latex_content += "\\subsection*{Maximization Step}\n"
        latex_content += "Updated Translation Probabilities:\n"
        latex_content += "\\begin{tabular}{lll}\n"
        latex_content += "\\hline\n"
        latex_content += "English Word & Spanish Word & $t(f|e)$ \\\\\n"
        latex_content += "\\hline\n"
        t = data['t']
        for f in t:
            for e in t[f]:
                prob = t[f][e]
                latex_content += f"{e} & {f} & {prob:.4f} \\\\\n"
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"

    latex_content += "\\end{document}"
    return latex_content
