import reflex as rx

config = rx.Config(
    app_name="bumblebee_studio",
    telemetry_enabled=False,
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(appearance="dark", accent_color="amber", radius="large"),
        ),
    ],
)
