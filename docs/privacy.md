# Privacy, corrections and removal

`athletevalue` names real people: college athletes, most of them students and some
of them 17 or 18 years old at enrolment. This page says what the project publishes
about them, what it does not, and how to get something corrected or removed.

## What the software outputs about a named player

- Ratings, wins and dollar figures computed from public box scores, public
  play-by-play and public federal filings. These are model estimates. None of them
  is a report of what a player is paid, and the software says so in every summary.
- The team table's "position among paid teammates" compares two model quantities
  against team medians. It is not a judgement about a person.

The software does not collect, store or transmit anything about the people who run
it, beyond the downloads it caches on their own machine.

## The deal registry

The registry is a CC BY 4.0 file of disclosed compensation deals, one row per deal,
with a public source for each. **It is closed to contributions** until the process
below is in place and staffed. The packaged file is empty.

When it opens, these rules apply:

1. A row needs a public URL that states the figure for that athlete. Figures
   attributed only to unnamed sources are not accepted.
2. Valuations published by rating products are model outputs, not deals, and are
   not accepted.
3. Records released by a school in response to a public-records request are
   accepted only if the release itself is public. Do not submit a document that
   appears to have been released in error.
4. Athletes must be 18 or older at the deal date.
5. Contributors certify that the figure is transcribed accurately from the cited
   page, that the page was public when submitted, and that they are not the
   athlete's agent or counterparty disclosing confidential terms.
6. Every row has a `status`: `active`, `disputed` or `withdrawn`. Disputed and
   withdrawn rows stay in the file as tombstones, with a `status_note`, and are
   never used in a fit. Deleting a row would not remove it from forks under CC BY 4.0;
   a tombstone travels with the data.

## Corrections and removal

If you are an athlete, a school, an agent or a journalist and a row is wrong,
disputed or something you want removed, open an issue at
https://github.com/jman4162/athletevalue/issues. If the request should not be
public, open an issue asking for a private contact and the maintainer will reply
with one. The maintainer will:

- acknowledge within 7 days;
- mark the row `disputed` on receipt of any credible objection, which removes it from
  every fit immediately;
- mark it `withdrawn` at the subject's request, with no requirement to prove the
  figure wrong;
- correct a transcription error against the cited source.

## Why FERPA is not the relevant law, and what is

FERPA governs education records held by institutions that receive federal
education funds. A figure a journalist published, or a school released under a
public-records law, is not an education record in this project's hands, and FERPA
creates no claim against a downstream republisher. The relevant exposure is state
privacy, publicity and defamation law, which is why the rules above are about
accuracy, public sourcing, age and the subject's ability to object.
