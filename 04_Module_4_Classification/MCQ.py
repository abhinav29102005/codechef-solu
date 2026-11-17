import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        b = await p.chromium.connect_over_cdp('http://127.0.0.1:9222')
        page = b.contexts[0].pages[0]
        
        # Click the radio button / choice for High Precision, Low Recall
        print("Selecting option...")
        await page.evaluate(r'''() => {
            const labels = Array.from(document.querySelectorAll('label, div, span, p'));
            const target = labels.find(el => el.innerText && el.innerText.trim() === 'High Precision, Low Recall');
            if (target) {
                const input = target.querySelector('input') || target.closest('label')?.querySelector('input') || target;
                input.click();
            }
        }''')
        await asyncio.sleep(1)
        
        # Click Submit
        print("Clicking Submit...")
        await page.evaluate(r'''() => {
            const btns = Array.from(document.querySelectorAll('button'));
            const sub = btns.find(b => b.innerText.trim() === 'Submit');
            if (sub) sub.click();
        }''')
        await asyncio.sleep(2)
        
        res = await page.evaluate(r'''() => {
            return document.body.innerText.slice(0, 1500);
        }''')
        for l in res.splitlines():
            if any(k in l.lower() for k in ['correct', 'wrong', 'passed', 'submit', 'precision']):
                print(">>", l)

if __name__ == '__main__':
    asyncio.run(main())