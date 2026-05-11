import { test } from 'node:test';
import assert from 'node:assert/strict';

// collectPageEvents 的纯逻辑副本（与 content.js 同款,便于测试）
function collectPageEvents(root) {
  const doc = root || globalThis;
  const rows = doc.querySelectorAll('.siderstockcalendarcontent table tbody tr');
  const events = [];
  for (const row of rows) {
    try {
      const timeEl = row.querySelector('.time');
      const eventEl = row.querySelector('.event');
      if (!timeEl || !eventEl) continue;
      const dateText = (timeEl.textContent || '').trim();
      if (!dateText) continue;
      const date = dateText.slice(0, 5);
      const firstChild = eventEl.firstElementChild;
      let type, rawTitle;
      if (firstChild) {
        type = (firstChild.textContent || '').trim();
        rawTitle = (eventEl.textContent || '').replace(firstChild.textContent || '', '').trim();
      } else {
        const text = (eventEl.textContent || '').trim();
        const idx = text.search(/[\s　]/);
        if (idx > 0) {
          type = text.slice(0, idx);
          rawTitle = text.slice(idx + 1).trim();
        } else {
          type = text;
          rawTitle = '';
        }
      }
      const hadDatePrefix = /^\d{4}年\d{2}月\d{2}日/.test(rawTitle);
      let title = rawTitle
        .replace(/^\d{4}年\d{2}月\d{2}日(?:发布)?[《【]?|[》】]/g, '')
        .replace(/^截止\d{4}年\d{2}月\d{2}日\s*/g, '')
        .trim();
      if (hadDatePrefix && type && title.endsWith(type)) title = title.slice(0, -type.length).trim();
      if (!title) title = rawTitle;
      events.push({ date, type, title });
    } catch (_) { /* 单行异常跳过 */ }
  }
  return events.slice(0, 10);
}

// -------- 辅助:构造 mock DOM --------

function mockEl(tag, classes, children, text) {
  const el = {
    tagName: tag.toUpperCase(),
    classList: { contains: (c) => classes.includes(c) },
    children: children || [],
    textContent: text || '',
    firstElementChild: (children && children.length > 0) ? children[0] : null,
    querySelector: function (sel) {
      for (const ch of this.children) {
        if (ch.classList && ch.classList.contains(sel.replace('.', ''))) return ch;
      }
      const deeper = this.children.reduce((acc, c) => acc || (c.querySelector && c.querySelector(sel)), null);
      return deeper;
    },
  };
  return el;
}

function mockRoot(rows) {
  return {
    querySelectorAll: function (sel) {
      if (sel === '.siderstockcalendarcontent table tbody tr') return rows;
      return [];
    },
  };
}

function mockRow(timeText, eventChildren, eventText) {
  const timeEl = mockEl('td', ['time'], [], timeText);
  const eventEl = mockEl('td', ['event'], eventChildren || [], eventText || '');
  const row = {
    children: [timeEl, eventEl],
    querySelector: function (sel) {
      if (sel === '.time') return timeEl;
      if (sel === '.event') return eventEl;
      return null;
    },
  };
  return row;
}

// -------- 测试 --------

test('collectPageEvents: 正常解析', () => {
  const rows = [
    mockRow('04-30\n2026', null, '研报 2026年04月30日发布《茅台业绩双降》研报'),
    mockRow('04-29', null, '公告 关于召开业绩说明会的公告'),
  ];
  const root = mockRoot(rows);
  const events = collectPageEvents(root);
  assert.equal(events.length, 2);
  assert.deepEqual(events[0], { date: '04-30', type: '研报', title: '茅台业绩双降' });
  assert.deepEqual(events[1], { date: '04-29', type: '公告', title: '关于召开业绩说明会的公告' });
});

test('collectPageEvents: 空容器返回 []', () => {
  const root = mockRoot([]);
  assert.deepEqual(collectPageEvents(root), []);
});

test('collectPageEvents: td.time 或 td.event 缺失时跳过', () => {
  // 缺 event
  const rowNoEvent = {
    children: [mockEl('td', ['time'], [], '04-30')],
    querySelector: (sel) => sel === '.time' ? mockEl('td', ['time'], [], '04-30') : null,
  };
  const root = mockRoot([rowNoEvent]);
  const events = collectPageEvents(root);
  assert.equal(events.length, 0);
});

test('collectPageEvents: 截断到 10 条', () => {
  const rows = [];
  for (let i = 0; i < 15; i++) {
    rows.push(mockRow(`04-${String(30 - i).padStart(2, '0')}`, null, `公告 测试公告${i + 1}`));
  }
  const root = mockRoot(rows);
  const events = collectPageEvents(root);
  assert.equal(events.length, 10);
});

test('collectPageEvents: 有子元素时用 firstElementChild 提取 type', () => {
  const typeChild = mockEl('span', ['type'], [], '分红');
  const descChild = mockEl('span', ['desc'], [], '2025年度分红方案实施');
  const row = mockRow('04-25', [typeChild, descChild]);
  const root = mockRoot([row]);
  const events = collectPageEvents(root);
  assert.equal(events.length, 1);
  assert.equal(events[0].type, '分红');
});

test('collectPageEvents: 单行异常不中断整体', () => {
  // 一个正常行 + 一个会抛异常的行（timeEl 为 null 时跳过）
  const badRow = {
    querySelector: () => { throw new Error('boom'); },
  };
  const goodRow = mockRow('04-20', null, '公告 正常公告');
  const root = mockRoot([badRow, goodRow]);
  const events = collectPageEvents(root);
  assert.equal(events.length, 1);
  assert.equal(events[0].date, '04-20');
});
