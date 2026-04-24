import { ChatApp } from './components/ChatApp'
import { AuthBridge } from './lib/AuthBridge'
import { I18nProvider } from './lib/I18nContext'

export default function App() {
  return (
    <I18nProvider>
      <AuthBridge />
      <ChatApp />
    </I18nProvider>
  )
}
