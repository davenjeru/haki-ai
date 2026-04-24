import { expect, type Locator, type Page } from '@playwright/test'

/**
 * Page Object for the left ThreadSidebar. Signed-in specs use the thread
 * list + rename affordance; signed-out specs assert the "sign in to save
 * your chats" prompt.
 */
export class ThreadSidebarPage {
  readonly page: Page
  readonly aside: Locator
  readonly newChatButton: Locator
  readonly signInPrompt: Locator
  readonly collapsedToggle: Locator

  constructor(page: Page) {
    this.page = page
    this.aside = page.getByRole('complementary', { name: /chat history|historia ya mazungumzo/i })
    this.newChatButton = this.aside.getByRole('button', { name: /new chat|mazungumzo mapya/i })
    this.signInPrompt = this.aside.getByText(/sign in to keep|ingia ili kuhifadhi/i)
    this.collapsedToggle = page.getByRole('button', { name: /expand sidebar|fungua/i })
  }

  row(threadId: string): Locator {
    return this.page.getByTestId(`thread-row-${threadId}`)
  }

  async expectVisible(): Promise<void> {
    await expect(this.aside).toBeVisible()
  }

  async startRename(threadId: string): Promise<void> {
    const row = this.row(threadId)
    await row.hover()
    await row.getByRole('button', { name: /rename chat|badilisha jina/i }).click()
  }

  async fillTitle(text: string): Promise<void> {
    const input = this.page.getByRole('textbox', { name: /chat title|jina la mazungumzo/i })
    await input.fill(text)
    await input.press('Enter')
  }
}
