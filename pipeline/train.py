import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from pipeline import compute_loss, get_score_distributions, test_model
from tqdm.auto import tqdm


# from IPython.display import clear_output


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_model(
    model,
    train_batch_generator,
    val_batch_generator,
    opt,
    n_epochs,
    ckpt_name=None,
    visualize=True,
):
    """
    Run training: forward/backward pass using train_batch_generator and evaluation using val_batch_generator.
    Log performance using loss monitoring and score distribution plots for validation set.
    """

    train_loss, val_loss = [], [1]
    val_loss_idx = [0]
    # best_model = None
    top_val_accuracy = 0

    for epoch in range(n_epochs):
        start_time = time.time()

        # Train phase
        model.train(True)  # enable dropout / batch_norm training behavior
        for (X_batch, y_batch) in tqdm(
            train_batch_generator, desc="Training", leave=False
        ):

            # move data to target device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            data_batch = {"img": X_batch, "label": y_batch}

            opt.zero_grad()

            # train on batch: compute loss, calc grads, perform optimizer step and zero the grads
            loss, model = compute_loss(model, data_batch)

            # compute backward pass
            loss.backward()

            opt.step()

            # log train loss
            train_loss.append(loss.detach().cpu().numpy())

        # Evaluation phase
        metric_results = test_model(model, val_batch_generator, subset_name="val")
        metric_results = get_score_distributions(metric_results)

        # if visualize:
        #     clear_output()

        # Logging
        val_loss_value = np.mean(metric_results["loss"])
        val_loss_idx.append(len(train_loss))
        val_loss.append(val_loss_value)

        if visualize:
            # tensorboard for the poor
            fig = plt.figure(figsize=(15, 5))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)

            ax1.plot(train_loss, color="b", label="train")
            ax1.plot(val_loss_idx, val_loss, color="c", label="val")
            ax1.legend()
            ax1.set_title("Train/val loss.")

            ax2.hist(
                metric_results["scores_0"],
                bins=50,
                range=[0, 1.01],
                color="r",
                alpha=0.7,
                label="cats",
            )
            ax2.hist(
                metric_results["scores_1"],
                bins=50,
                range=[0, 1.01],
                color="g",
                alpha=0.7,
                label="dogs",
            )
            ax2.legend()
            ax2.set_title("Validation set score distribution.")

            plt.show()

        print(
            "Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time
            )
        )
        # train_loss_value = np.mean(train_loss[-n_train // BATCH_SIZE :])
        val_accuracy_value = metric_results["accuracy"]
        if val_accuracy_value > top_val_accuracy and ckpt_name is not None:
            top_val_accuracy = val_accuracy_value

            with open(ckpt_name, "wb") as f:
                torch.save(model, f)

    return model, opt
