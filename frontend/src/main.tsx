import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ClerkProvider } from '@clerk/clerk-react'
import './index.css'
import App from './App.tsx'

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

if (!PUBLISHABLE_KEY) {
  // Surfaces fast in dev if the env var is missing; in production builds we
  // let the provider itself raise so the error is still actionable.
  console.warn(
    'VITE_CLERK_PUBLISHABLE_KEY is not set — sign-in features will be unavailable.',
  )
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ClerkProvider
      publishableKey={PUBLISHABLE_KEY}
      appearance={{
        variables: {
          colorPrimary: '#40916c',
          colorBackground: '#141a18',
          colorInputBackground: '#0a0d0c',
          colorText: '#c8f0d8',
          colorTextSecondary: '#8a9a94',
          colorInputText: '#c8f0d8',
          colorNeutral: '#8a9a94',
          borderRadius: '12px',
        },
      }}
    >
      <App />
    </ClerkProvider>
  </StrictMode>,
)
