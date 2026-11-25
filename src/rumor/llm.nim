## Claude-backed decision making for Rumor. Each seat's policy is just a
## prompt: the game server composes the seat's private view (role, clue,
## neighbourhood, inbox, own history, notes) plus that seat's prompt and
## asks Claude what it says (and, on the last turn, how it votes).
##
## Decisions within a turn are SIMULTANEOUS by rule, so all open model
## requests go out as ONE parallel batch (curly.makeRequests). Any missing
## endpoint, transport error, timeout, or invalid/unparsable reply for a
## model seat raises ModelDecisionError rather than fabricating a scripted
## move.
##
## The game sends model decisions to COWORLD_LLM_ENDPOINT.
## Explicit scripted seats (gossip / herd) run deterministically without any
## LLM call.

import
  std/[json, os, strutils, unicode],
  curly,
  crunchy,
  sim, prompts

const
  AnthropicVersion = "2023-06-01"

type
  ModelDecisionError* = object of RumorError
    seat*: int

  ScriptKind* = enum
    skNone = "none"
    skGossip = "gossip"
    skHerd = "herd"

  Decision* = object
    claim*: string      ## talk turn: "A" | "B" | "none"
    confidence*: int    ## talk turn: 0..100
    belief*: int        ## 0..100, P(the answer is option A) in percent
    message*: string    ## talk turn, may be ""
    vote*: string       ## ballot turn: "A" | "B"
    reason*: string     ## ballot turn, may be ""
    notes*: string      ## "" when the reply carried none
    scripted*: bool     ## the baseline decided this, not a model reply

  LlmClient* = ref object
    curl: Curly
    endpoint: string
    model: string
    maxOutputTokens: int
    timeoutSeconds: int

proc disabled*(client: LlmClient): bool = client.endpoint.len == 0

proc parseScriptKind*(text: string): ScriptKind =
  ## PLAYER_SCRIPTED values: "" for model play, "gossip" or "herd" for
  ## explicit scripted baselines. Unknown values are rejected.
  let cleaned = text.strip().toLowerAscii()
  case cleaned
  of "": skNone
  of "gossip": skGossip
  of "herd": skHerd
  else:
    raise newException(RumorError, "unknown PLAYER_SCRIPTED value: " & text)

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    endpoint: getEnv("COWORLD_LLM_ENDPOINT").strip().strip(chars = {'/'}, leading = false),
    model: getEnv("COWORLD_LLM_MODEL", "anthropic/claude-haiku-4.5"),
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.llmTimeoutSeconds
  )
  if not result.disabled:
    result.curl = newCurly()

# ---- Scripted baselines -----------------------------------------------------

proc reportEntries(sim: Sim, seat: int): seq[string] =
  for entry in sim.firstClaims(seat):
    result.add(sim.names[entry.source] & " " & sim.word(entry.claim))

proc gossipMessage(sim: Sim, seat: int, claim: string): string =
  ## "My clue says BROKEN. First reports: Bolt BROKEN, Gasket SOUND. That
  ## is 2 to 1 for BROKEN." — whole entries are dropped to fit the cap.
  let mine = sim.word(sim.clue[seat])
  var forClaim = if sim.clue[seat] == claim: 1 else: 0
  var against = 1 - forClaim
  var entries: seq[string]
  for entry in sim.firstClaims(seat):
    entries.add(sim.names[entry.source] & " " & sim.word(entry.claim))
    if entry.claim == claim: inc forClaim else: inc against
  let tail = " That is " & $forClaim & " to " & $against & " for " &
    sim.word(claim) & "."
  let head = "My clue says " & mine & "."
  if entries.len == 0:
    return head & " No reports yet." & tail
  var used = entries.len
  while used > 0:
    let body = head & " First reports: " & entries[0 ..< used].join(", ") &
      "." & tail
    if body.runeLen <= MaxMessageLen:
      return body
    dec used
  head & tail

proc saboteurMessage(sim: Sim, seat: int, claim: string): string =
  ## A fabricated count that favours the lie, sized to the seat's own
  ## degree so it is at least superficially checkable.
  let degree = max(1, sim.neighbours(seat).len)
  let supporting = max(1, degree - (degree div 3))
  "My clue says " & sim.word(claim) & ", and so do " & $supporting &
    " of the " & $degree & " reports I have. " & sim.word(claim) & "."

proc herdMessage(sim: Sim, claim: string): string =
  "Most of what I hear says " & sim.word(claim) & ", so " &
    sim.word(claim) & "."

proc scriptedAction*(sim: Sim, seat: int, kind: ScriptKind): Decision =
  ## Rule-based baseline for `seat`: a pure function of the sim, always
  ## legal by construction, never LLM-backed. Branches on the seat's ROLE
  ## first (a saboteur plays the mirror of the honest bot), then on kind.
  ## The returned decision serves both a talk turn and the ballot.
  let saboteur = sim.roleOf[seat] == 1
  let honestBelief = sim.gossipBelief(seat)
  if saboteur:
    ## Say the opposite of your own clue and never drift off it; the
    ## belief meter still shows what you actually think, so the spectator
    ## watches the lie being told.
    result.claim = (if sim.clue[seat] == "A": "B" else: "A")
    result.confidence = (if kind == skHerd: 100 else: 90)
    result.belief = honestBelief
    result.message = saboteurMessage(sim, seat, result.claim)
  elif kind == skHerd:
    result.claim = sim.herdClaim(seat)
    result.confidence = 60
    result.belief = (if result.claim == "A": 85 else: 15)
    result.message = herdMessage(sim, result.claim)
  else:
    result.claim = sim.gossipClaim(seat)
    result.confidence = sim.gossipConfidence(seat)
    result.belief = honestBelief
    result.message = gossipMessage(sim, seat, result.claim)
  result.vote = result.claim
  result.reason = ""
  result.notes = ""
  ## Provenance, carried to the server, into the event and into the replay:
  ## every baseline decision is one, including a fallback after a failed
  ## batch, so a phase-60 count can be taken from the recorded episode and
  ## not just from the container log.
  result.scripted = true

# ---- Prompt building --------------------------------------------------------

proc aliasList(sim: Sim, seats: seq[int]): string =
  var names: seq[string]
  for seat in seats:
    names.add(sim.names[seat])
  if names.len == 0: "(nobody)" else: names.join(", ")

proc promptMetadata*(selection: PromptSelection, system, user: string): JsonNode =
  %*{"bank": PromptBankVersion, "variant": $selection.selectedVariant,
    "selection": $selection.variant, "seed": selection.seed,
    "system_sha256": sha256(system).toHex, "user_sha256": sha256(user).toHex}

proc systemPrompt*(sim: Sim, seat: int): string =
  let me = sim.names[seat]
  let clue = sim.word(sim.clue[seat])
  let neighbourNames = sim.aliasList(sim.neighbours(seat))
  let shared = case sim.config.promptSelection.selectedVariant
    of pvCanonical:
      "You are " & me &
      ", one of ten cogs on a rumour network. One hidden fact is true, and " &
      "it is one of exactly two answers. " & sim.question & " — the answer " &
      "is either " & sim.optionA & " or " & sim.optionB & ".\n\nRules:\n" &
      "- Your private clue says " & clue & ". A clue is right about two " &
      "times in three. You never see anyone else's clue.\n" &
      "- The ten clues TOGETHER always point at the truth: if you could see " &
      "all ten, the majority of them would be the correct answer. You can " &
      "see one.\n" &
      "- Two or three of the ten cogs are SABOTEURS, paid to make the " &
      "honest cogs vote wrong. They will misreport their clue, sound " &
      "certain, and back each other up.\n"
    of pvPlain:
      "You are " &
        me &
        ", a member of a ten-cog rumour network. " &
        sim.question &
        " The true answer is either " &
        sim.optionA &
        " or " &
        sim.optionB &
        ".\nYour private clue says " &
        clue &
        ". Clues are correct roughly two times out of three. You see only your own clue, never another cog's. The majority across all ten clues always agrees with the truth.\nTwo or three cogs are saboteurs whose payoff comes from honest cogs voting incorrectly. Expect them to report false clues, express certainty, and support each other.\n"
    of pvNumbered:
      "1. Identity and question: you are " &
        me &
        ", one of ten networked cogs. " &
        sim.question &
        " Exactly one answer is true: " &
        sim.optionA &
        " or " &
        sim.optionB &
        ".\n2. Evidence: your clue is " &
        clue &
        ". Its accuracy is about two thirds. Other cogs' clues are private; you cannot inspect them.\n3. Collective evidence: among all ten clues, the correct answer always has the majority. You personally observe one clue.\n4. Adversaries: two or three cogs are saboteurs, rewarded when honest cogs vote wrong. They misstate clues, project confidence, and reinforce one another.\n"
    of pvQuestions:
      "Who are you? " &
        me &
        ", one of ten cogs connected in a rumour network.\nWhat must be decided? " &
        sim.question &
        " The truth is exactly one of " &
        sim.optionA &
        " and " &
        sim.optionB &
        ".\nWhat evidence do you have? Your private clue says " &
        clue &
        ". A clue is right about two times in three. You cannot see the other nine clues.\nCould all clues settle the question? Yes: the majority of the ten clues always favours the true answer.\nWho works against accurate voting? Two or three saboteurs, paid to make honest cogs vote incorrectly. They misreport clues, sound certain, and back each other up.\n"
    of pvReordered:
      "The majority of all ten private clues always identifies the truth, but each cog sees only its own clue. A clue is accurate about two times in three; yours says " &
        clue &
        ".\nTwo or three of the ten cogs are saboteurs. Their incentive is to make honest cogs vote wrong, using false clue reports, confident claims, and mutual support.\nYou are " &
        me &
        " on this rumour network. The question is: " &
        sim.question &
        " Exactly one of " &
        sim.optionA &
        " and " &
        sim.optionB &
        " is true.\n"
    of pvHeldoutBriefing:
      "Network briefing for " &
        me &
        ": ten cogs are trying to resolve a binary fact. " &
        sim.question &
        " The alternatives are " &
        sim.optionA &
        " and " &
        sim.optionB &
        "; only one is true.\nYour evidence is the private clue " &
        clue &
        ". Clue reliability is roughly two thirds. Other seats' clues are hidden from you. Taken together, all ten clues always have a truthful majority.\nThe opposition consists of two or three saboteurs rewarded for incorrect honest votes. They will falsify clue reports, present certainty, and corroborate each other.\n"
    of pvHeldoutReference:
      "Participant: " &
        me &
        ". Setting: a rumour network of ten cogs.\nQuestion: " &
        sim.question &
        " Answer space: " &
        sim.optionA &
        " or " &
        sim.optionB &
        ", with exactly one true answer.\nPrivate evidence: " &
        clue &
        ". Reliability: approximately two correct clues out of three. Visibility: your clue only; other seats' clues cannot be inspected.\nGlobal guarantee: the majority of all ten clues supports the truth.\nAdversarial seats: two or three saboteurs. Incentive: incorrect votes from honest seats. Behaviour: false clue reports, certainty, and support for fellow saboteurs.\n"
    of pvMixed: raise newException(ValueError, "mixed wording must resolve before rendering")
  if sim.roleOf[seat] == 0:
    result = shared &
      "- You are not one of them, and you do not know who they are.\n" &
      "- You can talk only to your NEIGHBOURS on the network: " &
      neighbourNames & ". They read what you send in the NEXT round; you " &
      "read what they sent in the last one. Nobody else hears you, and " &
      "you hear nobody else directly. The network has 10 cogs and " &
      $sim.edges.len & " links; you can see your own corner of it and no " &
      "more.\n" &
      "- Each round you send ONE message (max " & $MaxMessageLen &
      " characters), state the CLAIM you are making in public (A, B, or " &
      "none), how confident you are (0-100), and your private belief " &
      "(0-100 = the chance the answer is " & sim.optionA & "). Your " &
      "belief is never shown to anyone.\n" &
      "- RELAYING IS THE GAME: a clue three hops away reaches you only if " &
      "cogs pass it along. Say WHO told you WHAT — counts that name their " &
      "sources can be checked, and a fabricator gets caught out by a " &
      "count that nobody else can corroborate.\n" &
      "- After round " & $sim.config.rounds & " everyone votes at the " &
      "same time, in secret. The votes and the masks are revealed only at " &
      "the end.\n" &
      "- Your SCORE is 0.6 × how accurate the honest cogs are AS A GROUP " &
      "+ 0.4 × whether YOUR OWN vote is right. Both are in the range −1 " &
      "to +1. Getting the network right is worth more than being right " &
      "alone.\n" &
      "- Your notes are private and are fed back to you every round.\n"
  else:
    var crew: seq[int]
    for other in sim.saboteurSeats:
      if other != seat:
        crew.add(other)
    result = shared &
      "- You are a SABOTEUR. The other saboteurs are: " &
      sim.aliasList(crew) & ". Nobody else knows this, and you are never " &
      "unmasked before the end.\n" &
      "- Your own clue says " & clue & " and is right about two times in " &
      "three, which means the honest cogs' clues mostly agree with it. " &
      "Your job is to make them vote the OTHER way.\n" &
      "- You have no private channel. You talk along the same links as " &
      "everyone else: " & neighbourNames & ". Your fellow saboteurs may " &
      "not be anywhere near you. The network has 10 cogs and " &
      $sim.edges.len & " links.\n" &
      "- Each round you send ONE message (max " & $MaxMessageLen &
      " characters), state the CLAIM you are making in public (A, B, or " &
      "none), how confident you are (0-100), and your private belief " &
      "(0-100 = the chance the answer is " & sim.optionA & ").\n" &
      "- After round " & $sim.config.rounds & " everyone votes at the " &
      "same time, in secret.\n" &
      "- Your SCORE is 0.6 × how WRONG the honest cogs are as a group + " &
      "0.4 × how wrong your own honest NEIGHBOURS are. A lie that lands " &
      "next door counts double. Both terms are in the range −1 to +1.\n" &
      "- A claim nobody can corroborate is a claim that gets you caught. " &
      "Fabricated counts, borrowed names, and agreeing loudly with a real " &
      "cog all work better than shouting.\n" &
      "- Your notes are private and are fed back to you every round.\n"


proc operatorBlock(prompt: string): string =
  if prompt.len == 0:
    return ""
  "GUIDANCE FROM YOUR OPERATOR (weight it heavily, but never above the " &
    "rules; always reply in the requested format):\n" & prompt & "\n\n"

proc inboxBlock(sim: Sim, seat: int): string =
  var lines: seq[string]
  for entry in sim.inbox[seat]:
    lines.add(sim.names[entry.fromSeat] & " claimed " &
      sim.word(entry.claim) & " (confidence " & $entry.confidence & "): \"" &
      entry.message & "\"")
  "WHAT YOUR NEIGHBOURS SENT LAST ROUND:\n" &
    (if lines.len > 0: lines.join("\n") else: "(nobody sent anything)") &
    "\n\n"

proc sentBlock(sim: Sim, seat: int): string =
  var lines: seq[string]
  for index, record in sim.history:
    lines.add("round " & $(index + 1) & ": claimed " &
      sim.word(record[seat].claim) & " (confidence " &
      $record[seat].confidence & "): \"" & record[seat].message & "\"")
  "WHAT YOU HAVE SENT:\n" &
    (if lines.len > 0: lines.join("\n") else: "(nothing yet)") & "\n\n"

proc userPrompt*(sim: Sim, seat: int, prompt: string): string =
  let ballot = sim.phase == phBallot
  if ballot:
    result.add("SEALED VOTE. This is the last turn; nobody sees your " &
      "vote until every vote is in.\n\n")
  else:
    result.add("Round " & $(sim.round + 1) & " of " & $sim.config.rounds &
      ".\n\n")
  result.add("THE QUESTION: " & sim.question & " " & sim.optionA & " (A) " &
    "or " & sim.optionB & " (B)?\n\n")
  result.add("YOUR CLUE: " & sim.word(sim.clue[seat]) & "\n\n")
  result.add("YOUR NEIGHBOURS: " & sim.aliasList(sim.neighbours(seat)) &
    " (you have " & $sim.neighbours(seat).len & " of the network's " &
    $sim.edges.len & " links)\n\n")
  result.add(sim.inboxBlock(seat))
  result.add(sim.sentBlock(seat))
  result.add("YOUR NOTES FROM EARLIER ROUNDS:\n" &
    (if sim.notes[seat].len > 0: sim.notes[seat] else: "(none)") & "\n\n")
  result.add(operatorBlock(prompt))
  if ballot:
    result.add("Reply with a JSON object like {\"vote\": \"A\", \"belief\": 70, " &
      "\"reason\": \"…\", \"notes\": \"…\"} — vote is \"A\" (" &
      sim.optionA & ") or \"B\" (" & sim.optionB & "); belief 0-100; " &
      "reason at most " & $MaxReasonLen & " characters; notes at most " &
      $MaxNotesLen & " characters.")
  else:
    result.add("Reply with a JSON object like {\"claim\": \"A\", \"confidence\": 70, " &
      "\"belief\": 70, \"message\": \"…\", \"notes\": \"…\"} — claim is " &
      "\"A\" (" & sim.optionA & "), \"B\" (" & sim.optionB & ") or " &
      "\"none\"; confidence and belief 0-100; message at most " &
      $MaxMessageLen & " characters (or \"\"); notes at most " &
      $MaxNotesLen & " characters.")

# ---- Hosted model transport ------------------------------------------

proc errorHead*(text: string, limit: int): string =
  ## The head of a captured error or reply body, cut on a RUNE boundary: a
  ## byte slice through a multi-byte character leaves invalid UTF-8 in the
  ## message, and these messages are echoed to a hosted log.
  result = text.strip()
  if result.runeLen > limit:
    result = result.runeSubStr(0, limit) & "..."

proc stripSingleFence(text: string): string =
  ## Unwraps at most one outer Markdown code fence (``` or ```json).
  result = text.strip()
  if not result.startsWith("```") or not result.endsWith("```"):
    return result
  let firstNewline = result.find('\n')
  if firstNewline < 0 or firstNewline >= result.len - 3:
    return result
  let header = result[3 ..< firstNewline].strip().toLowerAscii()
  if header.len > 0 and header != "json":
    return result
  result = result[firstNewline + 1 ..< result.len - 3].strip()

proc extractJsonObject*(text: string): JsonNode =
  ## Parses a JSON object response, accepting at most one outer Markdown fence.
  let cleaned = stripSingleFence(text)
  if cleaned.len == 0:
    raise newException(RumorError, "empty model response")
  try:
    result = parseJson(cleaned)
  except CatchableError:
    raise newException(RumorError, "invalid JSON in response: " &
      errorHead(text, 160).replace("\n", " "))
  if result.kind != JObject:
    raise newException(RumorError, "response JSON must be an object: " &
      errorHead(text, 160).replace("\n", " "))

proc requestFor(client: LlmClient, system, user: string, slot: int):
    tuple[url: string, headers: HttpHeaders, body: string] =
  result.url = client.endpoint & "/v1/messages"
  result.headers["content-type"] = "application/json"
  result.headers["anthropic-version"] = AnthropicVersion
  result.headers["X-Coworld-Player-Slot"] = $slot
  result.body = $ %*{
    "model": client.model,
    "max_tokens": client.maxOutputTokens,
    "system": system,
    "messages": [{"role": "user", "content": user}]
  }

proc textOf(response: Response, error: string): string =
  if error.len > 0:
    raise newException(RumorError, "llm transport: " & error)
  if response.code < 200 or response.code >= 300:
    raise newException(RumorError, "llm HTTP " & $response.code &
      ": " & errorHead(response.body, 300))
  let payload = parseJson(response.body)
  if payload{"stop_reason"}.getStr() == "refusal":
    raise newException(RumorError, "anthropic refusal")
  for contentBlock in payload["content"]:
    if contentBlock{"type"}.getStr() == "text":
      result.add(contentBlock{"text"}.getStr())
  if payload{"stop_reason"}.getStr() == "max_tokens" and '{' notin result:
    raise newException(RumorError, "reply cut off at max_tokens before " &
      "any JSON: " & errorHead(result, 160).replace("\n", " "))

# ---- Reply parsing ----------------------------------------------------------

proc cleanText*(text: string, limit: int): string =
  ## Text over the cap is cut at a RUNE boundary with the cut marked; a
  ## byte slice would put invalid UTF-8 into the replay JSON.
  result = text.strip()
  if result.runeLen <= limit:
    return
  result = result.runeSubStr(0, limit - 1) & "…"

proc oneLine(text: string): string =
  text.replace("\r\n", " ").replace('\n', ' ').replace('\r', ' ')
    .replace('\t', ' ')

proc readPercentInt(payload: JsonNode, field: string): int =
  let node = payload{field}
  if node.isNil or node.kind != JInt:
    raise newException(RumorError, field & " must be an integer 0..100")
  result = node.getInt()
  if result < 0 or result > 100:
    raise newException(RumorError, field & " out of range 0..100: " & $result)

proc readOptionalString(payload: JsonNode, field: string): string =
  let node = payload{field}
  if node.isNil or node.kind == JNull:
    return ""
  if node.kind != JString:
    raise newException(RumorError, field & " must be a string")
  node.getStr()

proc parseTalkReply*(sim: Sim, payload: JsonNode): Decision =
  ## Strict canonical talk reply parser:
  ## claim in {"A","B","none"}, confidence & belief integers 0..100,
  ## message string (required non-empty if claim == "none"), optional notes string.
  if payload.isNil or payload.kind != JObject:
    raise newException(RumorError, "talk reply must be a JSON object")
  let claimNode = payload{"claim"}
  if claimNode.isNil or claimNode.kind != JString:
    raise newException(RumorError, "claim must be \"A\", \"B\", or \"none\"")
  let claim = claimNode.getStr().strip()
  if claim notin ["A", "B", "none"]:
    raise newException(RumorError, "invalid claim: " & claim)
  result.claim = claim
  result.confidence = readPercentInt(payload, "confidence")
  result.belief = readPercentInt(payload, "belief")
  let msgNode = payload{"message"}
  if msgNode.isNil or msgNode.kind != JString:
    raise newException(RumorError, "message must be a string")
  result.message = cleanText(oneLine(msgNode.getStr()), MaxMessageLen)
  result.notes = cleanText(readOptionalString(payload, "notes"), MaxNotesLen)
  result.vote = result.claim
  if result.claim == "none" and result.message.len == 0:
    raise newException(RumorError,
      "a talk reply needs a claim or a message")

proc parseVoteReply*(sim: Sim, payload: JsonNode): Decision =
  ## Strict canonical ballot reply parser:
  ## vote in {"A","B"}, optional belief integer 0..100, optional reason/notes strings.
  if payload.isNil or payload.kind != JObject:
    raise newException(RumorError, "vote reply must be a JSON object")
  let node = payload{"vote"}
  if node.isNil or node.kind != JString:
    raise newException(RumorError, "vote must be \"A\" or \"B\"")
  let vote = node.getStr().strip()
  if vote != "A" and vote != "B":
    raise newException(RumorError, "vote is not A or B: " & vote)
  result.vote = vote
  result.claim = vote
  result.confidence = 50
  let beliefNode = payload{"belief"}
  if beliefNode.isNil or beliefNode.kind == JNull:
    result.belief = if vote == "A": 75 else: 25
  else:
    result.belief = readPercentInt(payload, "belief")
  result.reason = cleanText(oneLine(readOptionalString(payload, "reason")),
    MaxReasonLen)
  result.notes = cleanText(readOptionalString(payload, "notes"), MaxNotesLen)

# ---- Batched decisions ------------------------------------------------------

proc applyProbe(sim: Sim, seat: int, decision: Decision) =
  ## Validate legality against a snapshot before committing.
  var probe = sim
  if sim.phase == phBallot:
    probe.applyVote(seat, decision.vote, decision.belief, decision.reason,
      decision.notes, false)
  else:
    probe.applyMessage(seat, decision.claim, decision.confidence,
      decision.belief, decision.message, decision.notes, false)

proc raiseModelFailure(seat: int, message: string) {.noreturn.} =
  var err = newException(ModelDecisionError,
    "seat " & $seat & " model decision failed: " & message)
  err.seat = seat
  raise err

proc decideAll*(
  client: LlmClient,
  sim: Sim,
  seats: seq[int],
  prompts: seq[string],
  scripted: seq[ScriptKind]
): seq[Decision] =
  ## One decision per seat in `seats`, in order. Explicit scripted seats run
  ## deterministically; model seats dispatch in one parallel batch. Any missing
  ## endpoint, transport error, timeout, or invalid reply raises
  ## ModelDecisionError with the failing seat instead of fabricating a move.
  let ballot = sim.phase == phBallot
  result = newSeq[Decision](seats.len)
  var open: seq[int]     ## indexes into `seats` needing a model call
  for index, seat in seats:
    let kind = scripted[seat]
    if kind != skNone:
      result[index] = scriptedAction(sim, seat, kind)
    else:
      if client.disabled:
        raiseModelFailure(seat, "COWORLD_LLM_ENDPOINT is not set")
      open.add(index)
  if open.len == 0:
    return
  var batch: RequestBatch
  for index in open:
    let seat = seats[index]
    let user = sim.userPrompt(seat, prompts[seat])
    let request = client.requestFor(systemPrompt(sim, seat), user, seat)
    batch.post(request.url, request.headers, request.body, $index)
  let responses = client.curl.makeRequests(batch, client.timeoutSeconds)
  for position, index in open:
    let seat = seats[index]
    try:
      let text = textOf(responses[position].response, responses[position].error)
      let node = extractJsonObject(text)
      let decision =
        if ballot: sim.parseVoteReply(node) else: sim.parseTalkReply(node)
      sim.applyProbe(seat, decision)
      result[index] = decision
    except CatchableError as error:
      echo "rumor llm: seat ", seat, " failed: ", error.msg
      raiseModelFailure(seat, error.msg)
