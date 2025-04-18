import pandas as pd
import plotly.graph_objects as go
import numpy as np
import argparse

# Function to read and process the CSV data


def load_data(file_path):
    # Parse the CSV, assuming it's comma-separated
    df = pd.read_csv(file_path)

    # Convert timestamp to seconds for x-axis
    # Assuming timestamp is in nanoseconds
    df['seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9

    # Convert memory values to MiB if they're in bytes
    memory_columns = ['allocated', 'active', 'metadata', 'resident', 'mapped']
    for col in memory_columns:
        if df[col].max() > 10000:  # Simple heuristic to check if in bytes
            df[col] = df[col] / (1024 * 1024)  # Convert bytes to MiB

    return df

# Function to create the plot


def create_memory_plot(df, output_file='memory_usage.html'):
    # Create figure
    fig = go.Figure()

    # Add active memory (blue)
    fig.add_trace(go.Scatter(
        x=df['seconds'],
        y=df['active'],
        fill='tozeroy',
        fillcolor='rgba(0, 0, 255, 0.5)',
        line=dict(color='blue', width=1),
        name='Total Allocated Memory'
    ))

    # Add allocated memory (red)
    fig.add_trace(go.Scatter(
        x=df['seconds'],
        y=df['allocated'],
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.5)',
        line=dict(color='red', width=1),
        name='Active Memory (in use)'
    ))

    # Update layout
    fig.update_layout(
        title='Memory Usage Over Time',
        xaxis_title='Time (s)',
        yaxis_title='Memory (MiB)',
        legend=dict(
            x=0.99,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)',
            bordercolor='rgba(0, 0, 0, 0.5)'
        ),
        hovermode='x unified',
        plot_bgcolor='white',
        height=600,
        width=1000
    )

    # Add grid lines
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        range=[0, max(df['allocated'].max(), df['active'].max()) * 1.2]
    )

    # Save the figure
    fig.write_html(output_file)

    return fig


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate memory usage plot from CSV data')
    parser.add_argument('-i', '--input', default='tmp/memory_stats.csv',
                        help='Input CSV file path (default: tmp/memory_stats.csv)')
    parser.add_argument('-o', '--output', default='memory_usage.html',
                        help='Output HTML file path (default: memory_usage.html)')

    # Parse arguments
    args = parser.parse_args()

    print(f"Loading data from {args.input}")
    df = load_data(args.input)

    fig = create_memory_plot(df, args.output)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()