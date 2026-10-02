import asyncio
import flet as ft
from typing import Dict, Tuple
from block_a import BlockA
from block_b import BlockB
from block_c import BlockC
from block_d import BlockD
from block_e import BlockE

def numfield(label: str, value: str, tooltip: str = "") -> ft.TextField:
    return ft.TextField(
        label=label,
        value=value,
        width=240,
        text_align=ft.TextAlign.RIGHT,
        helper="0 .. 1",
        tooltip=tooltip or "Enter a number between 0 and 1",
        keyboard_type=ft.KeyboardType.NUMBER,
        dense=True,
        filled=True,
        border_radius=12,
    )

def parse_inputs(vals: Dict[str, ft.TextField]) -> Tuple[bool, Dict[str, float], str]:
    out = {}
    try:
        for k, tf in vals.items():
            v = float((tf.value or "").strip())
            if not (0.0 <= v <= 1.0):
                return False, {}, f"'{tf.label}' must be between 0 and 1."
            out[k] = v
        return True, out, ""
    except ValueError:
        return False, {}, "Please enter valid numeric values (0..1)."

def main(page: ft.Page):
    page.title = "Distillation Tray Fuzzy Model"
    page.padding = 16
    page.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
    page.theme_mode = ft.ThemeMode.LIGHT
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window.min_width = 1000
    page.window.min_height = 720
    page.scroll = ft.ScrollMode.AUTO

    def snack(msg: str):
        page.show_dialog(ft.SnackBar(ft.Text(msg)))

    title_row = ft.Row(
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        controls=[
            ft.Row(
                spacing=12,
                controls=[
                    ft.Icon(ft.Icons.SCIENCE_ROUNDED, size=28),
                    ft.Text("Distillation Tray Fuzzy Model", weight=ft.FontWeight.BOLD, size=22),
                ],
            ),
            ft.IconButton(
                icon=ft.Icons.DARK_MODE_ROUNDED,
                tooltip="Toggle light/dark theme",
                on_click=lambda e: toggle_theme(),
            ),
        ],
    )

    def toggle_theme():
        page.theme_mode = (
            ft.ThemeMode.DARK if page.theme_mode == ft.ThemeMode.LIGHT else ft.ThemeMode.LIGHT
        )
        page.update()

    tf_HV_k     = numfield("Vapor hold up at step k", "0.5")
    tf_HL_k     = numfield("Liquid hold up at step k", "0.5")
    tf_P_bot    = numfield("Bottom pressure of the tray", "0.33")
    tf_P_top    = numfield("Top vapor pressure of the tray", "0.02")
    tf_Qlin_k   = numfield("Liquid inlet at step k", "0.5")
    tf_Qvin_k   = numfield("Vapor inlet at step k", "0.5")
    tf_TLin_k   = numfield("Temperature of liquid inlet", "0.6")
    tf_TLbulk_k = numfield("Temperature of bulk liquid", "0.2")
    tf_TVin_k   = numfield("Temperature of vapor inlet", "0.6")
    tf_TVbulk_k = numfield("Temperature of bulk vapor", "0.2")

    inputs_map: Dict[str, ft.TextField] = {
        "HV_k": tf_HV_k,
        "HL_k": tf_HL_k,
        "P_bot": tf_P_bot,
        "P_top": tf_P_top,
        "Qlin_k": tf_Qlin_k,
        "Qvin_k": tf_Qvin_k,
        "TLin_k": tf_TLin_k,
        "TLbulk_k": tf_TLbulk_k,
        "TVin_k": tf_TVin_k,
        "TVbulk_k": tf_TVbulk_k,
    }
    defaults = {key: tf.value for key, tf in inputs_map.items()}

    card_geometry = ft.Card(
        elevation=6,
        content=ft.Container(
            padding=16,
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.Text("Holdups / Pressures", weight=ft.FontWeight.BOLD),
                    ft.ResponsiveRow(
                        columns=12,
                        controls=[
                            ft.Container(tf_HV_k, col={"xs": 12, "sm": 6, "md": 3}),
                            ft.Container(tf_HL_k, col={"xs": 12, "sm": 6, "md": 3}),
                            ft.Container(tf_P_bot, col={"xs": 12, "sm": 6, "md": 3}),
                            ft.Container(tf_P_top, col={"xs": 12, "sm": 6, "md": 3}),
                        ],
                    ),
                ],
            ),
        ),
    )

    card_flows = ft.Card(
        elevation=6,
        content=ft.Container(
            padding=16,
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.Text("Inlets (Volumetric Flows)", weight=ft.FontWeight.BOLD),
                    ft.ResponsiveRow(
                        columns=12,
                        controls=[
                            ft.Container(tf_Qlin_k, col={"xs": 12, "sm": 6, "md": 6}),
                            ft.Container(tf_Qvin_k, col={"xs": 12, "sm": 6, "md": 6}),
                        ],
                    ),
                ],
            ),
        ),
    )

    card_temps = ft.Card(
        elevation=6,
        content=ft.Container(
            padding=16,
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.Text("Temperatures", weight=ft.FontWeight.BOLD),
                    ft.ResponsiveRow(
                        columns=12,
                        controls=[
                            ft.Container(tf_TLin_k,   col={"xs": 12, "sm": 6, "md": 3}),
                            ft.Container(tf_TLbulk_k, col={"xs": 12, "sm": 6, "md": 3}),
                            ft.Container(tf_TVin_k,   col={"xs": 12, "sm": 6, "md": 3}),
                            ft.Container(tf_TVbulk_k, col={"xs": 12, "sm": 6, "md": 3}),
                        ],
                    ),
                ],
            ),
        ),
    )

    out_hl = ft.Text("", size=16)
    out_hv = ft.Text("", size=16)
    out_ql = ft.Text("", size=16)
    out_qv = ft.Text("", size=16)

    result_card = ft.Card(
        elevation=8,
        content=ft.Container(
            padding=16,
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.PRIMARY),
            border_radius=16,
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Row(
                        [ft.Icon(ft.Icons.INSIGHTS, color=ft.Colors.PRIMARY), ft.Text("Results", weight=ft.FontWeight.BOLD)],
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    out_hl,
                    out_hv,
                    out_ql,
                    out_qv,
                ],
            ),
        ),
    )

    run_btn = ft.FilledButton(
        "Run Fuzzy Model",
        icon=ft.Icons.PLAY_ARROW_ROUNDED,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
    )
    clear_btn = ft.OutlinedButton(
        "Clear",
        icon=ft.Icons.CLEAR_ALL,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)),
    )

    progress_text = ft.Text("", size=12, color=ft.Colors.ON_SURFACE_VARIANT)
    progress_bar = ft.ProgressBar(width=420, value=0, visible=False, color=ft.Colors.PRIMARY)

    MF_key = "Triangular"
    T_norm_key = "min"
    S_norm_key = "max"
    defuzz_method = "centroid"

    def do_clear(_=None):
        for key, tf in inputs_map.items():
            tf.value = defaults[key]
        out_hl.value = out_hv.value = out_ql.value = out_qv.value = ""
        page.update()

    async def do_run_async(_=None):
        if run_btn.disabled:
            return
        ok, vals, msg = parse_inputs(inputs_map)
        if not ok:
            snack(msg)
            return
        run_btn.disabled = True
        clear_btn.disabled = True
        progress_bar.visible = True
        progress_bar.value = 0
        progress_text.value = "Preparing..."
        page.update()
        for i in range(0, 60, 3):
            progress_bar.value = i / 100
            progress_text.value = f"Working... {i}%"
            page.update()
            await asyncio.sleep(0.02)
        try:
            HV_k     = vals["HV_k"]
            HL_k     = vals["HL_k"]
            P_bot    = vals["P_bot"]
            P_top    = vals["P_top"]
            Qlin_k   = vals["Qlin_k"]
            Qvin_k   = vals["Qvin_k"]
            TLin_k   = vals["TLin_k"]
            TLbulk_k = vals["TLbulk_k"]
            TVin_k   = vals["TVin_k"]
            TVbulk_k = vals["TVbulk_k"]
            Qvout_A, Qlout_A = await asyncio.to_thread(BlockA, MF_key, T_norm_key, S_norm_key, defuzz_method, HV_k, HL_k, P_bot, P_top)
            for i in range(60, 75, 3):
                progress_bar.value = i / 100
                progress_text.value = f"Working... {i}%"
                page.update()
                await asyncio.sleep(0.01)
            if P_bot < P_top:
                HL_k = 0
            else:
                HL_k = HL_k
            HL_k1_B          = await asyncio.to_thread(BlockB, MF_key, T_norm_key, S_norm_key, defuzz_method, Qlin_k, Qlout_A, HL_k)
            for i in range(75, 85, 3):
                progress_bar.value = i / 100
                progress_text.value = f"Working... {i}%"
                page.update()
                await asyncio.sleep(0.01)
            if P_bot < P_top:
                HV_k = 0
            else:
                HV_k = HV_k
            HV_k1_C          = await asyncio.to_thread(BlockC, MF_key, T_norm_key, S_norm_key, defuzz_method, Qvin_k, Qvout_A, HV_k)
            for i in range(85, 93, 3):
                progress_bar.value = i / 100
                progress_text.value = f"Working... {i}%"
                page.update()
                await asyncio.sleep(0.01)
            if TLin_k == TLbulk_k:
                HV_k1_D, HL_k1_D = HV_k1_C, HL_k1_B
            else:
                HV_k1_D, HL_k1_D = await asyncio.to_thread(BlockD, MF_key, T_norm_key, S_norm_key, defuzz_method, TLin_k, TLbulk_k, HV_k1_C, HL_k1_B)
            if TVin_k == TVbulk_k:
                HV_k1, HL_k1 = HV_k1_D, HL_k1_D
            else:
                HV_k1, HL_k1     = await asyncio.to_thread(BlockE, MF_key, T_norm_key, S_norm_key, defuzz_method, TVin_k, TVbulk_k, HV_k1_D, HL_k1_D)
            progress_bar.value = 1.0
            progress_text.value = "Finalizing..."
            page.update()
            out_hl.value = f"HL at k+1 = {HL_k1:.3f}"
            out_hv.value = f"HV at k+1 = {HV_k1:.3f}"
            out_ql.value = f"Qlout at k = {Qlout_A:.3f}"
            out_qv.value = f"Qvout at k = {Qvout_A:.3f}"
            page.update()
            await asyncio.sleep(0.15)
            snack("Fuzzy model computed successfully.")
        except Exception as ex:
            snack(f"Error while computing: {ex}")
        finally:
            progress_bar.visible = False
            progress_text.value = ""
            run_btn.disabled = False
            clear_btn.disabled = False
            page.update()

    run_btn.on_click = do_run_async
    clear_btn.on_click = do_clear

    page.add(
        title_row,
        ft.Container(height=8),
        ft.Row(
            controls=[card_geometry, card_flows, card_temps],
            wrap=True,
            spacing=16,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        ft.Container(height=8),
        ft.Row(
            [run_btn, clear_btn],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=16,
        ),
        ft.Container(height=6),
        ft.Column(
            controls=[progress_bar, progress_text],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        result_card,
        ft.Container(height=8),
        ft.Text(
            size=12,
            color=ft.Colors.ON_SURFACE_VARIANT,
            italic=True,
        ),
    )

if __name__ == "__main__":
    ft.run(main)
