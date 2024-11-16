import sys
import copy
from PyQt6.QtWidgets import QApplication
from ui_module import TranslationUI
from data_processing import preprocess_sentence
from em_algorithm import initialize_probabilities, em_step
from latex_output import generate_latex_output
import logging

logging.basicConfig(level=logging.INFO, filename='app.log',
                    format='%(asctime)s %(levelname)s:%(message)s')

# Example usage
logging.info("Training started.")


def start_training(ui):
    try:
        source_sentence = ui.source_input.text()
        target_sentence = ui.target_input.text()
        source_tokens = preprocess_sentence(source_sentence)
        target_tokens = preprocess_sentence(target_sentence)

        t = initialize_probabilities(source_tokens, target_tokens)
        iteration_data = []

        iteration_count = ui.iterations_input.value()
        for _ in range(iteration_count):
            t_copy = copy.deepcopy(t)
            t, count, norm, total_e, delta = em_step(
                t, source_tokens, target_tokens)
            iteration_data.append({
                't': t_copy,
                'count': count,
                'norm': norm,
                'total_e': total_e,
                'delta': delta,
            })

        latex_content = generate_latex_output(
            iteration_data, source_tokens, target_tokens)
        ui.latex_preview.setPlainText(latex_content)
        ui.save_button.setEnabled(True)
        ui.latex_content = latex_content
    except Exception as e:
        logging.error("Error during training: %s", str(e))
        ui.display_error(
            "An error occurred during training. Check the log file for details.")


def save_latex(ui):
    if hasattr(ui, 'latex_content'):
        file_name = "em_iterations.tex"
        with open(file_name, "w") as f:
            f.write(ui.latex_content)
        print(f"LaTeX output saved as '{file_name}'.")


def validate_sentences(source_sentence: str, target_sentence: str) -> bool:
    if not source_sentence.strip():
        ui.display_error("Source sentence cannot be empty.")
        return False
    if not target_sentence.strip():
        ui.display_error("Target sentence cannot be empty.")
        return False
    return True


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = TranslationUI()

    def on_train_button_clicked():
        source_sentence = ui.source_input.text()
        target_sentence = ui.target_input.text()
        if not validate_sentences(source_sentence, target_sentence):
            return
        try:
            start_training(ui)
        except ValueError as ve:
            ui.display_error(str(ve))

    def on_save_button_clicked():
        save_latex(ui)

    ui.train_button.clicked.connect(on_train_button_clicked)
    ui.save_button.clicked.connect(on_save_button_clicked)
    ui.show()
    sys.exit(app.exec())
