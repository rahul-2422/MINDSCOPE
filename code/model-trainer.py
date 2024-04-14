import pickle as pkl

from classification import metrics, multi_roc_plot, multi_roc_plot_mpl
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

if __name__ == "__main__":
    gnb = GaussianNB()
    lr = LogisticRegression(solver="liblinear", multi_class="auto")
    svc = SVC(
        gamma="auto",
        probability=True,
    )
    knn = KNeighborsClassifier(n_neighbors=27)
    xgb = XGBClassifier()
    etc = ExtraTreesClassifier(n_estimators=50)
    rfc = RandomForestClassifier(n_estimators=50)

    with open("../features-data/features_data.pkl", "rb") as f:
        data = pkl.load(f)

    with open("../channel-selection-data/chnl_selected_labels.pkl", "rb") as f:
        labels = pkl.load(f)

    metrics(gnb, data, labels)
    metrics(lr, data, labels)
    metrics(svc, data, labels)
    metrics(knn, data, labels)
    metrics(rfc, data, labels)
    metrics(etc, data, labels)
    metrics(xgb, data, labels)

    multi_roc_plot([rfc, etc, xgb], data, labels)

    multi_roc_plot_mpl([rfc, etc, xgb], data, labels)
