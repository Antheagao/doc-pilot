Your previous call to the `record_extraction` tool did not match the
required schema. The following paths were malformed or missing:

{{violations}}

Call `record_extraction` again. Every leaf in the schema -- each
top-level field, and each line item's `description`, `quantity`,
`unit_price`, and `total` -- must be an object of the exact form
`{"value": ..., "confidence": ...}`, where `confidence` is a number.
Fix ONLY the structure of the leaves listed above so they conform to
that shape. Do not re-read the document, and do not change the value
or confidence of any field you already got right -- keep everything
else exactly as it was in your previous call.
