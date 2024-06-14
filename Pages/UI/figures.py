import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from Utils.data_processing import backtest
from Utils.logger_config import configure_logger, shared_log_stream

logger = configure_logger('Figures', shared_log_stream)

def generate_figures_for_tickers(tickers):
    figures = {}
    for ticker in tickers:
        data = backtest(ticker)
        fig = generate_figure_for_ticker(ticker, *data)
        figures[ticker] = fig
    return figures

def generate_figure_for_ticker(ticker, dates, closes, cash_values, account_values, position_sizes, buys_x, buys_y, sells_x, sells_y, predictions, actual_movements, bt_accuracy, evaluation_metrics, feature_importances, preprocessed_data):
    
    # create plotly plot
    fig = make_subplots(rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Trading Data for {ticker}','Backtest Accuracy','Training Evaluation Metrics', 'Support Metrics', 'Feature Importances', 'Portfolio Value', 'Position Over Time'))

    # First plot (Trading Data)
    # Side note: The legend tag is simply to help with te alignment in fig.update_layout.
    fig.add_trace(go.Scatter(x=dates, y=np.array(closes).tolist(), mode='lines', name='Close Price', legend='legend1'),
                row=1, col=1)  # Plot close price on row 1 col 1
    fig.add_trace(go.Scatter(x=buys_x, y=buys_y, mode='markers', marker=dict(color='green', size=15), name='Buy Order',
                legend='legend1'), row=1, col=1)  # Plot the buys on row 1 col 1
    fig.add_trace(
        go.Scatter(x=sells_x, y=sells_y, mode='markers', marker=dict(color='red', size=15), name='Sell Order',
                legend='legend1'), row=1, col=1)   # Plot the sells on row 1 col 1
    
    # Second plot (Backtest Accuracy)
    fig.add_trace(go.Bar(x=['Backtest Accuracy'], y=[bt_accuracy], name='Backtest Accuracy', legend='legend2'), row=2, col=1)

    # Third Plot (Machine learning model Evaluation Metrics)
    metrics_keys, metrics_values = flatten_evaluation_metrics(evaluation_metrics['classification_report'])
    fig.add_trace(go.Bar(x=metrics_keys, y=metrics_values, name='Evaluation Metrics', legend='legend3'), row=3, col=1)
    
    # Fourth Plot (Support Metrics)
    support_keys, support_values = flatten_support_metrics(evaluation_metrics['classification_report'])
    fig.add_trace(go.Bar(x=support_keys, y=support_values, name='Support Metrics', legend='legend4'), row=4, col=1)

    # Fifth Plot (Feature Importances)
    feature_names = list(preprocessed_data.columns)
    fig.add_trace(go.Bar(x=feature_names, y=feature_importances, name='Feature Importances', legend='legend5'), row=5, col=1)

    # Sixth Plot (Portfolio Value)
    fig.add_trace(
        go.Scatter(x=dates, y=np.array(cash_values).tolist(), mode='lines', name='Cash Over Time', legend='legend6'),
        row=6, col=1)  # plot the cash value on row 6 col 1
    fig.add_trace(go.Scatter(x=dates, y=np.array(account_values).tolist(), mode='lines', name='Account Value Over Time',
        legend='legend6'), row=6, col=1)  # plot the account values on row 6 col 1

    # Seventh Plot (Position over Time)
    fig.add_trace(go.Scatter(x=dates, y=np.array(position_sizes).tolist(), mode='lines', name='Position Over Time',
        legend='legend7'), row=7, col=1)  # plot the position over time on row 7 col 1.

    # Update xaxis properties
    fig.update_layout(
        xaxis=dict(rangeselector=dict(buttons=list([dict(count=1, label="1m", step="month", stepmode="backward", ),
                                                    dict(count=6, label="6m", step="month", stepmode="backward"),
                                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                                    dict(count=1, label="1y", step="year", stepmode="backward"),
                                                    dict(step="all")])),
                   showline=True,  # Shows the x-axis line
                   showgrid=True,  # Shows the x-axis grid
                   showticklabels=True),  # Shows the x-axis tick labels
        xaxis2=dict(showline=True, showgrid=True, showticklabels=True),
        xaxis3=dict(showline=True, showgrid=True, showticklabels=True),
        height=2500,
        template="plotly_dark",
        legend1={"y": 1},
        legend2={"y": 0.77},
        legend3={"y": 0.56},
        legend4={"y": 0.33},
        legend5={"y": 0.10},
        legend6={"y": 0.05},
        xaxis_rangeselector_font_color='white',
        xaxis_rangeselector_activecolor='red',
        xaxis_rangeselector_bgcolor='black',
    )
    return fig

    # Flatten the evaluation metrics dictionary for easy plotting
def flatten_evaluation_metrics(classification_report):
    """Flatten the nested classification report into keys and values."""
    metrics_keys = []
    metrics_values = []

    for class_label, metrics in classification_report.items():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg', 'support']:
            for metric_name, metric_value in metrics.items():
                metrics_keys.append(f'{class_label} {metric_name}')
                metrics_values.append(metric_value)

    return metrics_keys, metrics_values

def flatten_support_metrics(classification_report):
    """Flatten support metrics into a dictionary for easy plotting."""
    support_keys = []
    support_values = []

    for class_label, metrics in classification_report.items():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            support_keys.append(f'{class_label} support')
            support_values.append(metrics['support'])

    return support_keys, support_values


