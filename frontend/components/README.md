# Components

## Important: this project is a static site, not a shadcn / React / TypeScript app

The EcoBudget frontend (`frontend/landing.html`) is plain static HTML, CSS, and
vanilla JavaScript. It has no bundler, no `package.json`, no Tailwind, and no
TypeScript. The shadcn `IPhoneMockup` React component cannot render in that page
as-is (a `.tsx` file needs a React build step).

So there are two things in this folder, and it is deliberate:

1. **`ui/iphone-mockup.tsx`** and **`ui/demo.tsx`** — the shadcn component exactly
   as provided, kept here at the conventional `components/ui/` path so it is ready
   to drop into a real React app later. `components/ui/` is the shadcn default
   import location (`@/components/ui/...`); keeping that exact path means the
   component and any future shadcn additions resolve without editing imports.

2. **A faithful vanilla port** of the same iPhone mockup is what the live
   `landing.html` actually renders (the `.iphone`, `.iphone-screen`,
   `.iphone-island` styles and the interactive "See it work" demo). It shows the
   real research idea end to end: pick a question, tap load, watch decompose →
   retrieve → stop → answer while a byte/energy meter compares task-sufficient
   loading against a full-page load, using the numbers measured on real pages.

## If you want to use the React `.tsx` version

The project would first need to become a React + Tailwind + TypeScript app:

```bash
# 1. scaffold a React + TypeScript app (Vite shown; Next.js also fine)
npm create vite@latest ecobudget-web -- --template react-ts
cd ecobudget-web && npm install

# 2. add Tailwind
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
# set content: ["./index.html","./src/**/*.{ts,tsx}"] in tailwind.config.js
# add the @tailwind base/components/utilities lines to src/index.css

# 3. initialise shadcn (creates components.json, sets up @/ alias + components/ui)
npx shadcn@latest init

# 4. copy iphone-mockup.tsx and demo.tsx into src/components/ui/
#    then use it:  import IPhoneMockup from "@/components/ui/iphone-mockup";
```

The `@/components/ui` alias is configured by `shadcn init` (it writes the path
alias into `tsconfig.json` and `components.json`). Placing the file anywhere else
would break the `@/components/ui/iphone-mockup` import that `demo.tsx` and shadcn
convention expect, which is why this folder mirrors that exact path.

`IPhoneMockup` has no required props (sensible defaults: `model="14-pro"`,
`color="space-black"`). Pass `children` to render your own screen content, or
`wallpaper` for a background image. No external assets or icons are required by
the component itself.
