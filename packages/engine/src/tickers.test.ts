import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  firstCanadianYahooWithBars,
  parseListEntry,
  pickCanonicalListing,
  resolveManualAgainstListings,
  type ParsedEntry,
} from './tickers.ts';

describe('parseListEntry Canadian listings', () => {
  it('maps TSX:RBY and RBY.TO to Yahoo RBY.TO', () => {
    assert.deepEqual(parseListEntry('TSX:RBY'), {
      yahoo: 'RBY.TO',
      tv: 'TSX:RBY',
      name: null,
    });
    assert.deepEqual(parseListEntry('RBY.TO'), {
      yahoo: 'RBY.TO',
      tv: 'RBY.TO',
      name: null,
    });
  });

  it('maps Google TSE:RBY to Yahoo RBY.TO and TradingView TSX:RBY', () => {
    assert.deepEqual(parseListEntry('TSE:RBY'), {
      yahoo: 'RBY.TO',
      tv: 'TSX:RBY',
      name: null,
    });
  });

  it('leaves a bare US ticker unchanged', () => {
    assert.deepEqual(parseListEntry('AAPL'), {
      yahoo: 'AAPL',
      tv: 'AAPL',
      name: null,
    });
  });
});

const rbyTo: ParsedEntry = {
  yahoo: 'RBY.TO',
  tv: 'TSX:RBY',
  name: 'Rubellite Energy Inc.',
};
const shopUs: ParsedEntry = {
  yahoo: 'SHOP',
  tv: 'NASDAQ:SHOP',
  name: 'Shopify Inc.',
};
const shopTo: ParsedEntry = {
  yahoo: 'SHOP.TO',
  tv: 'TSX:SHOP',
  name: 'Shopify Inc.',
};

describe('resolveManualAgainstListings', () => {
  it('resolves bare RBY to the TSX listing in the universe', () => {
    assert.deepEqual(resolveManualAgainstListings('RBY', [rbyTo]), rbyTo);
  });

  it('keeps a US ticker when the universe has the unsuffixed listing', () => {
    assert.deepEqual(resolveManualAgainstListings('SHOP', [shopUs, shopTo]), shopUs);
  });

  it('does not rewrite an explicit Canadian listing', () => {
    const typed = parseListEntry('RBY.TO');
    assert.deepEqual(resolveManualAgainstListings('RBY.TO', [rbyTo]), typed);
    assert.deepEqual(resolveManualAgainstListings('TSX:RBY', [rbyTo]), parseListEntry('TSX:RBY'));
  });

  it('keeps a bare ticker that is not in the universe', () => {
    assert.deepEqual(resolveManualAgainstListings('ZZZZ', [rbyTo]), {
      yahoo: 'ZZZZ',
      tv: 'ZZZZ',
      name: null,
    });
  });
});

describe('pickCanonicalListing', () => {
  it('prefers .TO over .V when both match the short symbol', () => {
    const v: ParsedEntry = { yahoo: 'RBY.V', tv: 'TSXV:RBY', name: null };
    assert.equal(pickCanonicalListing('RBY', [v, rbyTo])?.yahoo, 'RBY.TO');
  });
});

describe('firstCanadianYahooWithBars', () => {
  it('picks RBY.TO when the bare Yahoo symbol has no bars', () => {
    const picked = firstCanadianYahooWithBars('RBY', (ticker) => ticker === 'RBY.TO');
    assert.equal(picked, 'RBY.TO');
  });

  it('does not probe other suffixes when the user already typed .TO', () => {
    assert.equal(
      firstCanadianYahooWithBars('RBY.TO', (ticker) => ticker === 'RBY.V'),
      null,
    );
  });
});
