# MKLight.UITests

Opt-in tests that drive the built Windows app with real input: UI Automation (FlaUI.UIA3) and
mouse messages posted to the app's own windows. They are skipped unless `MKLIGHT_UI_TESTS=1`,
so a normal `dotnet test` of the solution passes over them in well under a second.

```powershell
cd apps/windows
dotnet build MKLight.slnx -p:Platform=x64
$env:MKLIGHT_UI_TESTS = '1'
dotnet test --project tests/MKLight.UITests/MKLight.UITests.csproj
```

- Each test launches `src/MKLight.App/bin/x64/Debug/.../MKLight.exe` (override with `MKLIGHT_APP`)
  with `--sample double_gauss --size 1600x1000 --theme light` and a scratch
  `MKLIGHT_SUPPORT_DIRECTORY`, and kills it at the end. Tests run one at a time (`xunit.runner.json`).
- On a failure the test saves a capture of the app window and its automation tree to
  `bin/.../ui-artifacts/` (or `MKLIGHT_UI_ARTIFACTS`).
- `ExploreTests.DumpTree` with `MKLIGHT_UI_DUMP=1` as well writes the whole automation tree and the
  app's child windows there: the way to find names while writing a test.
- The tests never change the user's settings: no theme switch, no save, no open-recent.

## The input rule: window-scoped only

These tests share the desktop with whoever is using the machine, so they never take over the
real mouse or keyboard and never look at anything but the app.

- **Allowed**: UI Automation on the app's elements: patterns (Invoke, Value, Toggle,
  ExpandCollapse, SelectionItem), SetFocus and ShowContextMenu; `PostMessage` of `WM_MOUSEMOVE`, `WM_LBUTTONDOWN/UP`,
  `WM_LBUTTONDBLCLK`, `WM_RBUTTONDOWN/UP`, `WM_MOUSEWHEEL` to the app's HWND or one of its child
  windows (`WindowMouse`), in that window's client coordinates computed from the element's UIA
  bounding rectangle; `PrintWindow` of the app's window (`WindowImage`) for screenshots and diffs.
- **Never**: `SendInput`, `mouse_event`, `keybd_event`, `SendKeys`, FlaUI's global `Mouse` and
  `Keyboard` classes, `SetCursorPos`, anything that types keys globally, or a capture of the
  screen. Text goes in through the ValuePattern, never as keystrokes.

WinUI 3 hosts a window's content in a `Microsoft.UI.Content.DesktopChildSiteBridge` child window;
`WindowMouse` posts there (the deepest app window under the point when there is no bridge).
The test process is per-monitor DPI aware (`DpiAwareness`), so automation rectangles, window
rects and message coordinates are all physical pixels, whatever the display scale.

Text fields commit when they lose the focus: a test focuses the field (UIA SetFocus), sets the
text with the ValuePattern, then moves the focus to another control (no Enter key).

## What WinUI 3 ignores

**Every window-targeted mouse message is ignored.** WinUI 3 (Windows App SDK 1.x) reads pointer
input from the system's pointer pipeline (`GetPointerInfo` on real input), not from the message
parameters. `ExploreTests.ProbeMouseTargets` posted and sent a click (`WM_MOUSEMOVE`,
`WM_LBUTTONDOWN`, `WM_LBUTTONUP`) and a synthetic `WM_POINTERENTER/UPDATE/DOWN/UP` on a sidebar
entry to each of the app's windows: the top-level `WinUIDesktopWin32WindowClass`,
`InputNonClientPointerSource`, `ReunionWindowingCaptionControls`,
`Microsoft.UI.Content.DesktopChildSiteBridge` and `InputSiteWindowClass`. None had any effect.
The same holds for clicks, double clicks (`WM_LBUTTONDBLCLK`), right clicks, drags with
`MK_LBUTTON` and `WM_MOUSEWHEEL`.

So the gestures are verified through UI Automation instead, and global input is never used as a
fallback:

| Gesture | Verified through |
|---|---|
| click on a sidebar entry | SelectionItem.Select |
| click on a lens row (select) | SelectionItem.Select on `LensRow.{i}` |
| double click on a lens cell, typing a radius | Invoke on `LensCell.{row}.radius` (starts editing), ValuePattern.SetValue on the editor, focus away; also ValuePattern on the cell itself |
| right click on a lens row → Make Stop / Insert | ShowContextMenu on the row (IUIAutomationElement3), Invoke on the menu item |
| drag of a pane's title onto another panel | the title's context menu (ShowContextMenu): Move to New Window |
| drag on the 3-D layout (orbit), wheel (zoom) | nothing yet: the camera has no automation surface |

The posted-mouse tests (`Posted*`) still run: each posts its gesture and, when nothing happens
within a few seconds, skips with the reason above and the name of the UI Automation test that
covers the feature. Should WinUI ever honour the messages, they start passing by themselves.

## Element lookups

All lookups live in `Ui.cs`, by AutomationId:

| Id | Element |
|---|---|
| `LensRow.{i}` | lens table row (0 is the object); SelectionItem |
| `LensCell.{row}.{key}` | a lens table cell; Value, Invoke starts editing |
| `LensEditor` | the in-place cell editor while a cell edits |
| `Panel.lens_data`, `Panel.{key}.Header` | the lens data panel's container; a panel's title (drag handle) |
| `Pane.{analysis}` and `.Header`, `.Title`, `.Run`, `.AutoUpdate`, `.Settings`, `.State`, `.More`, `.Backend` | an analysis pane and its parts |
| `Settings.{analysis}.{field}` | a settings form control (field: the setting's key path, e.g. `max_frequency`; a number with a unit is a group whose Edit holds the value) |
| `Sidebar.{analysis}`, `Sidebar.lens_data`, `Sidebar.layout`, `Sidebar.issues` | sidebar entries; SelectionItem |
| `AddAnalysis.{category}`, `AddAnalysis.{analysis}` | Add Analysis menu items (the Analysis menu's New Analysis Pane items use `NewPane.` instead) |
| `Status.EFL`, `Status.FNumber`, `Status.TT`, `Status.Squeeze`, `Status.Issues` | the status capsule's value texts (name = the text) and its issues button |
| `Layout.Canvas` | the layout; its name starts "Layout, meridional section" / "x–z section" / "3D view" and follows the mode |
| `Layout.Mode.section`, `.xz`, `.3d` | the view selector's items; SelectionItem (IsSelected = shown) |
| `Inspector.Tab.{id}`, `Inspector.Row.{key}` | inspector page buttons and rows |
| `UndoButton`, `RedoButton`, `AddAnalysisButton`, `NumbersButton`, `ScaleBar` | x:Names WinUI exposes as ids |

The inspector's keys are the engine's field paths, not the labels: the sphere's radius row is
`Inspector.Row.type.Shape.curvature` (its name is "Radius"). Still found another way: the inspector's
current page (by position under "Inspector pages") and the other menu items (by name).

