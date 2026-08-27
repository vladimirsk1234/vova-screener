import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { isFundamentalsPendingError, parseApiErrorBody } from './apiError.ts';

describe('parseApiErrorBody', () => {
  it('uses Nest message when the body is JSON', () => {
    assert.equal(
      parseApiErrorBody(
        404,
        '{"message":"No fundamentals in Mongo for ADBE yet — wait for the EOD refresh","error":"Not Found","statusCode":404}',
      ),
      'No fundamentals in Mongo for ADBE yet — wait for the EOD refresh',
    );
  });

  it('unwraps a "404 : {json}" proxy dump', () => {
    assert.equal(
      parseApiErrorBody(
        404,
        '404 : {"message":"Fundamentals for ADBE are still loading. Updating 12/4800.","error":"Not Found","statusCode":404}',
      ),
      'Fundamentals for ADBE are still loading. Updating 12/4800.',
    );
  });
});

describe('isFundamentalsPendingError', () => {
  it('treats EOD / still-loading text as retryable', () => {
    assert.equal(
      isFundamentalsPendingError(
        new Error('No fundamentals in Mongo for ADBE yet — wait for the EOD refresh'),
      ),
      true,
    );
    assert.equal(
      isFundamentalsPendingError(new Error('Fundamentals for ADBE are still loading. Updating 3/10.')),
      true,
    );
    assert.equal(isFundamentalsPendingError(new Error('Set FMP_API_KEY on the API server')), false);
  });

  it('does not treat hard missing-fundamentals errors as pending', () => {
    assert.equal(
      isFundamentalsPendingError(new Error('No fundamentals for BRK-B. No FMP fundamentals for BRK-B')),
      false,
    );
    assert.equal(
      isFundamentalsPendingError(new Error('No FMP fundamentals for BRK-B')),
      false,
    );
  });
});
