import sys
import copy
import logging
from typing import List, Dict
from fluency import train_language_model, calculate_fluency
from adequacy import calculate_adequacy
from bleu_score import calculate_bleu
from PyQt6.QtWidgets import QApplication, QMessageBox
from ui_module import TranslationUI
from data_processing import preprocess_sentence
from em_algorithm import initialize_probabilities, em_step
from latex_output import generate_latex_output


logging.basicConfig(level=logging.INFO, filename='app.log',
                    format='%(asctime)s %(levelname)s:%(message)s')

# Example usage
logging.info("Training started.")


def generate_translation(source_tokens: List[str], translation_probs: Dict) -> List[str]:
    """
    Generates translation using the trained probability table.
    """
    translation = []
    for source_word in source_tokens:
        # Find the target word with highest probability for this source word
        best_prob = 0
        best_word = None
        for target_word, probs in translation_probs.items():
            if probs[source_word] > best_prob:
                best_prob = probs[source_word]
                best_word = target_word
        translation.append(best_word if best_word else source_word)
    return translation


def start_training(ui: TranslationUI):
    try:
        source_sentence = ui.source_input.text()
        target_sentence = ui.target_input.text()
        iteration_count = ui.iterations_input.value()

        # Validation steps remain the same...
        source_tokens = preprocess_sentence(source_sentence)
        target_tokens = preprocess_sentence(target_sentence)

        t = initialize_probabilities(source_tokens, target_tokens)
        iteration_data = []

        # Training iterations remain the same...
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

        # Generate translation using trained model
        translated_tokens = generate_translation(source_tokens, t)
        translated_sentence = ' '.join(translated_tokens)

        # Fluency Score - using translated sentence
        target_corpus = ["la casa es pequeña", "el perro es grande"]
        language_model = train_language_model(target_corpus)
        fluency_score = calculate_fluency(translated_sentence, language_model)

        # Adequacy Score - comparing source and generated translation
        adequacy_score = calculate_adequacy(
            source_sentence, translated_sentence)

        # BLEU Score - comparing reference (target) with generated translation
        bleu_score = calculate_bleu(target_sentence, translated_sentence)

        # Add translation output to UI
        ui.translation_output.setText(translated_sentence)

        # Update scores
        ui.update_scores_display(
            fluency_score=fluency_score,
            adequacy_score=adequacy_score,
            bleu_score=bleu_score
        )

        # Generate LaTeX with translation results
        latex_content = generate_latex_output(
            iteration_data=iteration_data,
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            translated_tokens=translated_tokens,
            fluency_score=fluency_score,
            adequacy_score=adequacy_score,
            bleu_score=bleu_score
        )

        ui.latex_preview.setPlainText(latex_content)
        ui.save_button.setEnabled(True)
        ui.latex_content = latex_content

        logging.info(f"Training completed. Translation: {translated_sentence}")

    except Exception as e:
        logging.error(f"Training error: {str(e)}")
        QMessageBox.critical(ui, "Error", f"Training failed: {str(e)}")


def save_latex(ui: TranslationUI):
    if hasattr(ui, 'latex_content'):
        file_name = "em_iterations.tex"
        with open(file_name, "w") as f:
            f.write(ui.latex_content)
        logging.info(f"LaTeX output saved as '{file_name}'.")
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
