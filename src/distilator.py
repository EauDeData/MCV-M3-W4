import wandb
from wandb.keras import WandbMetricsLogger

import keras
import tensorflow as tf

class Distiller(keras.Model):

    # FROM OFFICIAL TUTORIAL: https://keras.io/examples/vision/knowledge_distillation/
    # AND PAPER: https://arxiv.org/abs/1503.02531

    def __init__(self, student, teacher, temperature):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.temperature = temperature

    def compile(self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1):

        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha

    def train_step(self, data):
        # Unpack data
        x, y = data
        y = tf.argmax(y, axis=-1)

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)

            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data
        y = tf.argmax(y, axis=-1)

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results


def train_student(
    config,
    student,
    trained_teacher,
    temperature,
    optimizer,
    train_set,
    test_set,
    metrics = [keras.metrics.SparseCategoricalAccuracy(), 'accuracy'],
    student_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    distill_loss = keras.losses.KLDivergence(),
    alpha = 0.1
):

    wandb.init(
        config=config, project='m3_week4',
        name=config["experiment_name"],
    )
    callbacks = [WandbMetricsLogger()]

    trainer = Distiller(student, trained_teacher, temperature)
    trainer.compile(
        optimizer=optimizer,
        metrics=metrics,
        student_loss_fn=student_loss,
        distillation_loss_fn=distill_loss,
        alpha=alpha,
    )
    trainer.fit(
        train_set,
        epochs=config["epochs"],
        validation_data=test_set,
        callbacks=callbacks,
        validation_freq=1,
    )
    trainer.evaluate(test_set)

    return student, trainer
