/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#0F172A',
          card: '#1E293B',
          accent: '#38BDF8',
          border: '#334155'
        },
        alert: {
          safe: '#10B981',      // Emerald 500
          warning: '#F59E0B',   // Amber 500
          critical: '#EF4444'   // Red 500
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      boxShadow: {
        'glow-safe': '0 0 15px rgba(16, 185, 129, 0.4)',
        'glow-warning': '0 0 15px rgba(245, 158, 11, 0.4)',
        'glow-critical': '0 0 20px rgba(239, 68, 68, 0.6)',
      }
    },
  },
  plugins: [],
}

