import { expect, test } from '../fixtures/test'
import { ChatPage } from '../pages/ChatPage'
import { ThreadSidebarPage } from '../pages/ThreadSidebarPage'

test.describe('Anonymous user', () => {
  test('can send a message and receive a citation', async ({ anonymousPage }) => {
    const chat = new ChatPage(anonymousPage.page)
    await chat.goto()

    await expect(chat.composer).toBeVisible()
    await chat.sendMessage('What are my rights if fired without notice?')
    await expect(chat.page.getByText(/Employment Act 2007/i).first()).toBeVisible()
    // The citations block is collapsed by default; expand it and assert the
    // first citation's section renders.
    await chat.page.getByRole('button', { name: /citations/i }).click()
    await expect(chat.page.getByText(/Sec\. 40/i).first()).toBeVisible()
  })

  test('sidebar shows the sign-in prompt, not the thread list', async ({ anonymousPage }) => {
    const sidebar = new ThreadSidebarPage(anonymousPage.page)
    await anonymousPage.page.goto('/')
    await sidebar.expectVisible()
    await expect(sidebar.signInPrompt).toBeVisible()
    // No user-facing thread rows are rendered for anonymous users.
    await expect(anonymousPage.page.getByRole('list', { name: /previous chats|mazungumzo ya awali/i })).toHaveCount(0)
  })

  test('header exposes Sign in when signed out', async ({ anonymousPage }) => {
    const chat = new ChatPage(anonymousPage.page)
    await chat.goto()
    await expect(chat.signInButton).toBeVisible()
  })

  test('new chat wipes the message thread', async ({ anonymousPage }) => {
    const chat = new ChatPage(anonymousPage.page)
    await chat.goto()
    await chat.sendMessage('Hi there')
    await expect(chat.page.getByText('Hi there').first()).toBeVisible()
    await chat.newChatButton.click()
    await expect(chat.page.getByText('Hi there')).toHaveCount(0)
  })

  test('threads API is never called while signed out', async ({ anonymousPage }) => {
    const chat = new ChatPage(anonymousPage.page)
    await chat.goto()
    await chat.sendMessage('Hi')
    await expect(chat.page.getByText('Hi').first()).toBeVisible()
    // page.route() handlers log calls; anonymous path must stay silent.
    expect(anonymousPage.mock.listCalls).toHaveLength(0)
    expect(anonymousPage.mock.claimCalls).toHaveLength(0)
    expect(anonymousPage.mock.renameCalls).toHaveLength(0)
  })
})
