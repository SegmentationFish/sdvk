# DESIGN.md — sdvk studio web UI

Direction:   quiet light instrument panel, hairline separation, single indigo accent
Mode/dials:  Operate · Expression 2 / Density 3 / Motion 1
Impact:      1 refined (micro-detail only: focus rings, tabular numerals, hairline dividers)
Reference:   none supplied — judged from the existing dark UI's information model; visual world replaced
Palette:     canvas #fbfbfa / surface #ffffff / sunken #f6f6f5
             text #141414 · text-2 #57575a · text-3 #6e6e71
             border #e7e7e5 · border-strong #d3d3d0
             accent #4a5cf5 (hover #3847e0) · on-accent #fff · accent-soft #eceefd
             danger #c03434 / #fbeeee · good #2f7a52 / #ebf5ef
Type:        system stack (-apple-system → Segoe UI → Microsoft YaHei UI); 13px body, 11.5px meta,
             two weights (400/550-600); mono only for paths and logs
Shape:       radius 5 (controls) / 7 (containers inside stage) / 10 (cards); 1px hairline borders only,
             no shadows except the canvas image
Density:     inputs & buttons 30px tall, primary 34px, card padding 16px, rail gap 20px
Asset rule:  no images but the generated output; icons authored inline SVG at stroke 1.6/16px
Signature:   none at this impact level — the dotted idle canvas is the only ornament

## Motion

Tokens:      --dur-fast 120ms / --dur-base 200ms / --dur-slow 320ms
             --ease-out cubic-bezier(.16,1,.3,1)   (entrances, expansions)
             --ease-in  cubic-bezier(.4,0,1,1)     (exits)
             --ease-in-out cubic-bezier(.65,0,.35,1) (loops)
Rules:       transform + opacity only; every UI acknowledgement within 120–200ms
             press = translateY(1px) at 90ms; nothing animates layout properties
Inventory:   sheet slide (300ms ease-out in / ease-in out) · backdrop fade 200ms
             tab indicator cubic-bezier(.34,1.28,.64,1) 280ms with 32ms content cross-in
             panel cross-fade on tab change (pane-in 220ms, translateY 6px)
             progress sheen 1.4s linear loop while running · good pill dot pulse 2.4s
             image entrance 420ms (opacity + translateY 8px + scale .992)
             drag-to-dismiss follows the pointer 1:1, threshold 92px, no transition while dragging
Reduced:     prefers-reduced-motion collapses all durations to .01ms; sheet panel/backdrop/indicator
             drop transitions entirely

## Responsive

≤1240px:     2 columns (params | canvas), output rail wraps to a full-width 3-up strip
≤900px:      canvas-first single column. All three panes move into a bottom sheet mounted on <body>;
             a fixed 44px toolbar holds 参数 / 提示词 / 生成. Layout height is pinned to
             100dvh − topbar − 68px so the canvas fills the viewport and the page never scrolls.
             Sheet: three panes share one scroll container, switched by data-view on .sheet.
Verification: 390 / 768 / 1100 / 1440 — scrollWidth equals viewport at every breakpoint,
             zero console errors, pane display state asserted per tab.
