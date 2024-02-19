from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interactive, IntSlider, fixed
import numpy as np

def get_metrics(model, X_train, X_test, y_train, y_test, roc = None, nn = None, cutoff = None):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    if nn:
        train_proba = y_pred_train
        test_proba = y_pred_test
        y_pred_train = pd.DataFrame(train_proba)[0].apply(lambda x: 1 if x>=cutoff else 0)
        y_pred_test = pd.DataFrame(test_proba)[0].apply(lambda x: 1 if x>=cutoff else 0)

    # Train metrics
    print('-'*20,'Train','-'*20)
    acc = accuracy_score(y_train, y_pred_train)
    prec = precision_score(y_train, y_pred_train)
    rec = recall_score(y_train, y_pred_train)
    auc_train = roc_auc_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    conf_mat_train = confusion_matrix(y_train, y_pred_train)
    print(f'Accuracy: {round(acc,4)}')
    print(f'Precision: {round(prec,4)}')
    print(f'Recall: {round(rec,4)}')
    print(f'AUC: {round(auc_train,4)}')
    print(f'F1 score: {round(f1,4)}')
    
  
    print('-'*20,'Test','-'*20)
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    auc_test = roc_auc_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    conf_mat_test = confusion_matrix(y_test, y_pred_test)
    print(f'Accuracy: {round(acc,4)}')
    print(f'Precision: {round(prec,4)}')
    print(f'Recall: {round(rec,4)}')
    print(f'AUC: {round(auc_test,4)}')
    print(f'F1 score: {round(f1,4)}')
    
    if roc:
        fig, axs = plt.subplots(1,3, figsize = (22,5))
        axs = axs.flatten()
        
        sns.heatmap(conf_mat_train, annot = True, fmt=".0f", ax = axs[0])
        sns.heatmap(conf_mat_test, annot = True, fmt=".0f", ax = axs[1])
        
        if nn:
            fpr_train, tpr_train, thresholds = roc_curve(y_train, train_proba)
            fpr_test, tpr_test, thresholds = roc_curve(y_test, test_proba)
        else:
            fpr_train, tpr_train, thresholds = roc_curve(y_train, model.predict_proba(X_train)[:,1])
            fpr_test, tpr_test, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
            
        
        axs[2].plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'AUC Train = {auc_train:.4f}' )
        axs[2].plot(fpr_test, tpr_test, color='darkblue', lw=2, label=f'AUC Test = {auc_test:.4f}' )
        axs[2].plot([0,1],[0,1], color = 'red')
        axs[2].set_title('ROC curve')
        axs[2].grid()
        plt.legend(loc="lower right")
        
    else:
        fig, axs = plt.subplots(1,2, figsize = (15,5))
        axs = axs.flatten()
        sns.heatmap(conf_mat_train, annot = True, fmt=".0f", ax = axs[0])
        sns.heatmap(conf_mat_test, annot = True, fmt=".0f", ax = axs[1])
        plt.show()
    axs[0].set_title('Train set')  
    axs[1].set_title('Test set')


def create_mask(A, width, height):
    A = list(map(int, A))
    a, b, w, h = A[0], A[1], A[2], A[3]
    mask = np.zeros((height, width))  
    mask[b:b+h, a:a+w] = 1  
    return mask

def plot_image(data, masks, idx):
    imagen = data[idx]
    fig, ax = plt.subplots(figsize = (10,10))
    ax.imshow(imagen, cmap='gray')
    ax.imshow(masks[idx], alpha=0.3, origin='lower')
    plt.show()
    
