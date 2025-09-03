#!/bin/perl
use File::Spec;

my $dir = $ARGV[0];
my @lines= `grep -ro "\\[.*\\](.*)" $dir --include "*md"`;
my $errors_found = 0;

sub disp {
  my ($filename, $text, $link) = @_;
  my $filename = File::Spec->abs2rel($filename, $dir);
  print "\e[31mIn $filename: bad link to [$text]:\e[0m\n$link\n\n";
}

sub url_exists {
  `curl --silent --head --fail "@_" 2> /dev/null`;
  return $? eq 0;
}

foreach my $item (@lines) {
  my ($filename, $result) = split /:/, $item, 2;
  if ($result =~ /\[((?:[^\[\]]+|\[[^\[\]]*\])*)\]\(([^)]+)\)/) {
    my $link_text = $1;
    my $link_url  = $2;
    if ($link_url =~ /^http/) {
      if (not url_exists($link_url)) {
        disp($filename, $link_text, $link_url);
        $errors_found = 1;
      }
    } else {
      if (not -e "$dir/$link_url") {
        disp($filename, $link_text, $link_url);
        $errors_found = 1;
      }
    }
  }
}

exit($errors_found);
