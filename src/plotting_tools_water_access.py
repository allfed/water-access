import plotly.express as px
import pandas as pd


def format_plotly_graphs(fig):
    fig.update_layout(
        # font_family="Droid Sans Mono”",
        # “Arial”, “Balto”, “Courier New”, “Droid Sans”, “Droid Serif”, “Droid Sans Mono”, “Gravitas One”, “Old Standard TT”, “Open Sans”, “Overpass”, “PT Sans Narrow”, “Raleway”, “Times New Roman”.
        font_color="dimgrey",
        title_font_color="dimgrey",
        title={"font": {"size": 30}},
        legend_title_font_color="dimgrey",
        # legend=dict(title="Legend"),
        # legend_traceorder="reversed",
        font=dict(size=16),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e6e6e6")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e6e6e6")

    # fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))

    return fig
