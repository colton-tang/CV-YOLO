/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Avenir Next', 'Segoe UI', 'sans-serif'],
      },
      boxShadow: {
        panel: '0 24px 70px rgba(12, 32, 61, 0.08)',
      },
    },
  },
  plugins: [],
}
