import { expect, type Locator, type Page } from '@playwright/test'

/**
 * Page Object for the main Haki AI chat UI (composer + thread).
 * Accessibility-first selectors are used throughout; any `data-testid`
 * read here also exists in the component so the assertion doubles as a
 * contract between test + component.
 */
export class ChatPage {
  readonly page: Page
  readonly heading: Locator
  readonly composer: Locator
  readonly sendButton: Locator
  readonly newChatButton: Locator
  readonly signInButton: Locator

  constructor(page: Page) {
    this.page = page
    this.heading = page.getByRole('heading', { level: 1, name: /Haki AI/i })
    this.composer = page.getByRole('textbox', { name: /question|swali/i })
    this.sendButton = page.getByRole('button', { name: /send|tuma/i })
    this.newChatButton = page
      .getByRole('button', { name: /new chat|mazungumzo mapya/i })
      .first()
    this.signInButton = page.getByRole('button', { name: /^sign in$|^ingia$/i }).first()
  }

  async goto(): Promise<void> {
    await this.page.goto('/')
    await expect(this.heading).toBeVisible()
  }

  async sendMessage(text: string): Promise<void> {
    await this.composer.fill(text)
    await this.sendButton.click()
  }

  messageByText(text: string | RegExp): Locator {
    return this.page.getByText(text).first()
  }

  /** First visible assistant reply — used when the mock response text is known. */
  async waitForAssistantReply(): Promise<Locator> {
    const reply = this.page
      .getByText(/Employment Act 2007|msaidizi|I can only help/i)
      .first()
    await reply.waitFor({ state: 'visible' })
    return reply
  }
}
