from django.http import JsonResponse
from .Preprocessors.rfc_preprocessor import RFCPreprocessor
from .MLModels.random_forest_model import RandomForestTrainer
from .Utils.data_processing import backtest
from .Utils.logger_config import configure_logger

logger = configure_logger(__name__)

# Create your views here.
def backtest_view(request, ticker):
    try:
        # Call the backtest function with the provided ticker
        dates, closes, cash_values, account_values, position_sizes, buys_x, buys_y, sells_x, sells_y, predictions, actual_movements, bt_accuracy, evaluation_metrics, feature_importances, preprocessed_data = backtest(ticker)

        data = {
            "dates": dates,
            "closes": closes,
            "cash_values": cash_values,
            "account_values": account_values,
            "position_sizes": position_sizes,
            "buys_x": buys_x,
            "buys_y": buys_y,
            "sells_x": sells_x,
            "sells_y": sells_y,
            "predictions": predictions,
            "actual_movements": actual_movements,
            "bt_accuracy": bt_accuracy,
            "evaluation_metrics": evaluation_metrics,
            "feature_importances": feature_importances,
            "preprocessed_data": preprocessed_data.to_dict(orient='records')
        }
        
        return JsonResponse(data, safe=False)

    except Exception as e:
        logger.error(f"Error in backtest_view: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)