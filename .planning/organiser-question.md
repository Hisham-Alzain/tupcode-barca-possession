# Question to the organisers (draft — the user sends it)

To: hackathon.talentarena@mobileworldcapital.com
Subject: More Than A Hack 2027 — rules question on adapting to the tracking's input characteristics

Hello,

Thank you for the earlier clarification that agents may read the raw data, and that training,
fitting or calibrating on the ground truth or on the provided video is not allowed (external data
only for training/tuning). We would like to confirm that two techniques are acceptable before we
submit:

1. Input-quality switch. While inspecting the provided tactical tracking (input only, never the
   ground truth), we noticed the ball track is much smoother than raw tracking: the 99th percentile
   of frame-to-frame ball acceleration is about 30–47 m/s², while every external dataset we use
   (Metrica, IDSSE) is 204–619 m/s². Our touch detector assumes a raw track, so we added a mode that
   switches on automatically when a match's ball track is low-passed (threshold 150 m/s²; any value
   between 48 and 204 gives identical output on every dataset). All parameters of that mode were
   chosen on external data only, using a test bench where we smooth the external matches' ball in
   the same way. No labels of the provided match were used.

2. Run-time self-adjustment. Our method estimates, for each match it is run on, the share of
   hidden-ball time that is a stoppage, from its own predictions (unsupervised EM, no labels), and
   adjusts its stoppage probabilities accordingly. It runs identically on any match; its design and
   gate were chosen on external data only.

Is it acceptable that (1) we designed a test bench and a mode based on input statistics of the
provided tracking, and (2) the algorithm adapts unsupervised to each match's own input at run
time? Both will be described in our technical document.

Thank you,
<TEAM NAME>
