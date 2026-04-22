import { ChatApp } from './components/ChatApp'
import { I18nProvider } from './lib/I18nContext'

export default function App() {
  return (
    <I18nProvider>
      <ChatApp />
    </I18nProvider>
  )
}
