import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_prediction_analysis(y_test, preds, test_meta, sample_city="Hồ Chí Minh", tail=200):
    """
    Visualizes model predictions through a time-series forecast for a specific city 
    and a global actual vs. predicted scatter plot.

    Args:
        y_test (pd.Series/np.array): Ground truth values.
        preds (np.array): Model predictions.
        test_meta (pd.DataFrame): Metadata containing 'city' and 'timestamp'.
        sample_city (str): The city name to visualize.
        tail (int): Number of recent hours to show in the forecast.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Forecast for a Sample City ---
    mask = test_meta["city"] == sample_city
    if mask.sum() > 0:
        city_dates = test_meta.loc[mask, "timestamp"]
        city_actual = np.array(y_test)[mask]
        city_preds = np.array(preds)[mask]

        axes[0].plot(city_dates[-tail:], city_actual[-tail:], label="Actual", color="black", alpha=0.6, linewidth=1.5)
        axes[0].plot(city_dates[-tail:], city_preds[-tail:], label="Predicted", color="red", linestyle="--", linewidth=1.5)
        axes[0].set_title(f"Forecast: {sample_city} (Last {tail} hours)")
        axes[0].set_ylabel(r"PM2.5 Concentration ($\mu g/m^3$)")
        axes[0].legend()
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
    else:
        axes[0].text(0.5, 0.5, f"City '{sample_city}' not found", ha="center", va="center", color="red")
        axes[0].set_title("Forecast Visualization Error")

    # --- Plot 2: Scatter Plot (Actual vs. Predicted) ---
    axes[1].scatter(y_test, preds, alpha=0.1, color="blue", label="Data Points")
    
    min_val = min(np.min(y_test), np.min(preds))
    max_val = max(np.max(y_test), np.max(preds))
    
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal ($y=x$)", linewidth=2)
    axes[1].set_xlabel("Actual PM2.5")
    axes[1].set_ylabel("Predicted PM2.5")
    axes[1].set_title("Scatter Plot: Error Distribution")
    axes[1].legend()
    axes[1].set_xlim([min_val, max_val])
    axes[1].set_ylim([min_val, max_val])

    plt.tight_layout()
    plt.show()

def plot_training_metrics(model, X_test, feature_names=None):
    """
    Visualizes internal model metrics: Learning Curve (if available) 
    and Feature Importance (or coefficients).

    Args:
        model: Trained model object (XGBoost, Random Forest, or Linear Regression).
        X_test (pd.DataFrame): Testing feature set used to extract default feature names.
        feature_names (list/np.array, optional): Explicit names (useful for One-Hot Encoded data).
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Learning Curve ---
    if hasattr(model, 'evals_result'):
        results = model.evals_result()
        if results and "validation_0" in results:
            metric = list(results["validation_0"].keys())[0]
            epochs = range(len(results["validation_0"][metric]))
            axes[0].plot(epochs, results["validation_0"][metric], label="Train Loss")
            if "validation_1" in results:
                axes[0].plot(epochs, results["validation_1"][metric], label="Validation Loss")
            axes[0].set_title(f"Learning Curve ({metric.upper()})")
            axes[0].set_xlabel("Epochs")
            axes[0].set_ylabel(metric.upper())
            axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "Learning Curve not available\n(Model does not provide evals_result)", 
                      ha="center", va="center", fontsize=12, color='gray')
        axes[0].set_title("Training Progress")

    # --- Plot 2: Feature Importance ---
    importances = None
    title = "Feature Importance"
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
        title = "Top 15 Features (Absolute Coefficients)"

    names = feature_names if feature_names is not None else X_test.columns
    
    if importances is not None:
        importances = np.ravel(importances)
        if len(importances) == len(names):
            df = pd.DataFrame({"Feature": names, "Importance": importances})
            df = df.sort_values(by="Importance", ascending=False).head(15)
            
            sns.barplot(x="Importance", y="Feature", data=df, ax=axes[1], palette="viridis", hue="Feature", legend=False)
            axes[1].set_title(title)
        else:
            axes[1].text(0.5, 0.5, "Feature names count mismatch", ha="center", va="center", color="red")
    else:
        axes[1].text(0.5, 0.5, "No importance/coefficient data found", ha="center")

    plt.tight_layout()
    plt.show()