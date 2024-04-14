import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def test_train_splitter(data, labels):
    fold = StratifiedKFold(n_splits=10)

    for train_index, test_index in fold.split(data, labels):
        x_train, x_test, y_train, y_test = (
            [data[i] for i in train_index],
            [data[i] for i in test_index],
            [labels[i] for i in train_index],
            [labels[i] for i in test_index],
        )

    return x_train, x_test, y_train, y_test


def plot_cnf(cnf_matrix, model):
    fig = px.imshow(cnf_matrix, color_continuous_scale="Blues")

    fig.update_layout(
        title="Confusion Matrix with Rest-case-labeled: 0, One-back-labeled: 1 for: "
        + str(model)[: str(model).find("(")],
        xaxis_title="Actual Labels",
        yaxis_title="Predicted Labels",
        width=700,
        height=700,
    )

    fig.update_layout(
        font=dict(size=12), xaxis=dict(tick0=0, dtick=1), yaxis=dict(tick0=0, dtick=1)
    )

    for i in range(len(cnf_matrix)):
        for j in range(len(cnf_matrix)):
            if cnf_matrix[i, j] >= 220:
                color = "white"
            else:
                color = "black"

            fig.add_annotation(
                text=str(cnf_matrix[i, j]),
                x=j,
                y=i,
                showarrow=False,
                font=dict(
                    color=color,
                    size=24,
                ),
            )

    fig.show()


def plot_roc(tpr, fpr, model):
    fig = px.line(
        x=fpr,
        y=tpr,
    )

    fig.add_scatter(
        x=[0, 1], y=[0, 1], line=dict(color="navy", dash="dash"), name="Guessing"
    )

    fig.update_layout(
        title="ROC Curve of : " + str(model)[: str(model).find("(")],
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700,
        height=500,
    )

    fig.show()


def model_trainer(model, data, labels):
    x_train, x_test, y_train, y_test = test_train_splitter(data, labels)

    # Training the models on x_train and y_train
    model.fit(x_train, y_train)

    # Getting the class-label predictions and class-label prediction probabilities from the trained model
    model_predictions = model.predict(
        X=x_test
    )  # model_predictions.shape = [len(x-test)], (label(ith epoch))

    y_pred_prob = model.predict_proba(
        X=x_test
    )  # y_pred_prob.shape = [len(x-test), 2], (prob(label0), prob(label1))

    return y_pred_prob[:, 1], model_predictions, y_test


def get_tpr_fpr_auc(y_pred_prob, y_test):
    # Getting the FalsePositiveRate and TruePositveRates from plotting the ROC curve
    fpr, tpr, _ = roc_curve(
        y_test,
        y_pred_prob,
    )

    auc = roc_auc_score(y_test, y_pred_prob)

    return tpr, fpr, auc


def cross_eval(model, data, labels):
    score = cross_val_score(
        model,
        data,
        labels,
        cv=10,
    )

    cross_val_acc = np.average(score)
    cross_val_std = np.std(score)

    return cross_val_acc, cross_val_std


def report_builder(y_test, model_predictions):
    classif_report = (
        pd.DataFrame(
            classification_report(
                y_true=y_test,
                y_pred=model_predictions,
                output_dict=True,
                zero_division=0,
            )
        )
        .drop(labels="accuracy", axis=1)
        .T.round(2)
    )

    return classif_report


def multi_roc_plot(models, data, labels):
    fpr_list = []
    tpr_list = []

    legend_list = []

    for model in models:
        y_pred_prob, model_predictions, y_test = model_trainer(model, data, labels)
        tpr, fpr, auc = get_tpr_fpr_auc(y_pred_prob, y_test)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

        index = str(model).find("(")
        legend_list.append(str(model)[:index])

    colors = px.colors.qualitative.Set1

    fig = px.line()

    for i, (tpr, fpr) in enumerate(zip(tpr_list, fpr_list)):
        fig.add_scatter(
            x=fpr, y=tpr, mode="lines", line=dict(color=colors[i]), name=legend_list[i]
        )

    fig.add_scatter(
        x=[0, 1], y=[0, 1], line=dict(color="navy", dash="dash"), name="Guessing"
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title=dict(text="<b>False Positive Rate</b>", font=dict(size=14)),
        yaxis_title=dict(text="<b>True Positive Rate</b>", font=dict(size=14)),
        width=700,
        height=500,
        legend=dict(
            x=1,
            y=0,
            traceorder="reversed",
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="Black",
            borderwidth=2,
        ),
    )

    fig.show()


def metrics(model, data, labels):
    # Using average of cross val score for accuracy
    cross_val_acc, cross_val_std = cross_eval(model, data, labels)

    # Training the model on x_train and y_train
    y_pred_prob, model_predictions, y_test = model_trainer(model, data, labels)

    # Building the Classification Report using the predictions as a dataframe without the accuracy column
    classif_report = report_builder(y_test, model_predictions)

    # Building the Confusion Matrix using the predicted class labels
    cnf_matrix = confusion_matrix(y_true=y_test, y_pred=model_predictions)

    tpr, fpr, auc = get_tpr_fpr_auc(y_pred_prob, y_test)

    print(
        f"\033[96m  \033[1m ---------------------------------------{str(model)[:str(model).find('(')]}---------------------------------------  \033[0m\n\n"
    )
    print(
        f"\033[91m Cross-val-Accuracy: {100*cross_val_acc:.2f}+-{cross_val_std:.2f} \033[0m \n"
    )

    print(f"\033[94m {classif_report} \033[0m \n")

    plot_cnf(cnf_matrix=cnf_matrix, model=model)

    plot_roc(tpr=tpr, fpr=fpr, model=model)

    print(
        f"\033[92m Area Under the ROC Curve (AUC) of {str(model)[:str(model).find('(')]}: {100*auc:.2f}\033[0m\n\n"
    )


def multi_roc_plot_mpl(models, data, labels):
    tprs = []
    fprs = []
    aucs = []

    fig, ax = plt.subplots()

    for i, model in enumerate(models):
        # Model predictions and ROC curve
        y_pred_prob, model_predictions, y_test = model_trainer(model, data, labels)
        tpr, fpr, auc = get_tpr_fpr_auc(y_pred_prob, y_test)

        tprs.append(tpr)
        fprs.append(fpr)
        aucs.append(auc)

        ax.plot(fpr, tpr, lw=2, color=f"C{i}")

    # Remove mean TPR plot

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="black")

    # Rest of code to set labels, legend, etc

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    ax.legend(
        labels=[
            f"RFC - AUC:{aucs[0]:.2f}",
            f"ETC - AUC:{aucs[1]:.2f}",
            f"XGB - AUC:{aucs[2]:.2f}",
        ],
        loc="lower right",
        fontsize=10,
    )

    fig.tight_layout()
    plt.show()
