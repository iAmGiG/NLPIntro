import sys
import copy
from PyQt6.QtWidgets import QApplication
from ui_module import TranslationUI
from data_processing import preprocess_sentence
from em_algorithm import initialize_probabilities, em_step
from latex_output import generate_latex_output


def start_training(ui):
    source_sentence = ui.source_input.text()
    target_sentence = ui.target_input.text()
    source_tokens = preprocess_sentence(source_sentence)
    target_tokens = preprocess_sentence(target_sentence)

    t = initialize_probabilities(source_tokens, target_tokens)
    iteration_data = []

    for _ in range(2):  # Two iterations
        t_copy = copy.deepcopy(t)
        t, count = em_step(t, source_tokens, target_tokens)
        iteration_data.append({'t': t_copy, 'count': count})

    latex_content = generate_latex_output(
        iteration_data, source_tokens, target_tokens)
    ui.latex_preview.setPlainText(latex_content)
    ui.save_button.setEnabled(True)
    ui.latex_content = latex_content  # Store for saving


def save_latex(ui):
    if hasattr(ui, 'latex_content'):
        file_name = "em_iterations.tex"
        with open(file_name, "w") as f:
            f.write(ui.latex_content)
        print(f"LaTeX output saved as '{file_name}'.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = TranslationUI()

    def on_train_button_clicked():
        start_training(ui)

    def on_save_button_clicked():
        save_latex(ui)

    ui.train_button.clicked.connect(on_train_button_clicked)
    ui.save_button.clicked.connect(on_save_button_clicked)
    ui.show()
    sys.exit(app.exec())
