def format_plotly_graphs(fig):
    fig.update_layout(
        # DejaVu Sans matches the matplotlib figures. It must be installed as a
        # system font for Kaleido's headless Chrome to resolve it; otherwise the
        # fallback is Verdana, whose heavier stems look darker than the
        # matplotlib output at the same colour and size.
        font_family="DejaVu Sans, Verdana, Arial, sans-serif",
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

    # fig.update_layout(
    #   legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    # )

    return fig
